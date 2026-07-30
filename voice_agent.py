"""Local uncensored voice agent — full-duplex with WebRTC AEC.

Pipeline: mic -> AEC -> Silero VAD -> Moonshine STT -> Ollama (SuperGemma4)
       -> sentence-stream -> Kokoro TTS -> AEC reverse -> speakers.

Echo cancellation is done via LiveKit's `AudioProcessingModule` (WebRTC AEC3).
The TTS output that we send to the speakers is also fed to the APM as a
reference signal. The APM subtracts the (delay-aligned) reference from the
mic signal so the agent does not hear its own voice. This lets us re-enable
true barge-in: the user can interrupt mid-reply and the agent will stop.

Set `HALF_DUPLEX=1` to disable AEC and fall back to mic-muted-while-talking.
"""
from __future__ import annotations

import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
import torch
from livekit.rtc import AudioFrame, AudioProcessingModule
from silero_vad import VADIterator, load_silero_vad

from engines import (
    SttEngine,
    TtsEngine,
    TurnDetector,
    build_stt,
    build_tts,
    build_turn_detector,
)

# ─────────────────────────────── config ───────────────────────────────

SR = 16000
APM_FRAME = 160          # 10 ms at 16 kHz — required by WebRTC APM
VAD_FRAME = 512          # silero-vad's required input size at 16 kHz
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M"
KEEP_ALIVE = "30m"

HALF_DUPLEX_ONLY = os.environ.get("HALF_DUPLEX", "0") == "1"
TTS_TAIL_GRACE_MS = 250  # grace period after TTS in half-duplex fallback
STREAM_DELAY_MS = int(os.environ.get("STREAM_DELAY_MS", "80"))  # spk→mic round-trip estimate

# Energy gate: while the agent is talking, ignore VAD trips whose RMS energy
# (on the AEC-cleaned signal) is below this floor. AEC residual on a Mac with
# built-in mic+speaker typically lands around 0.005–0.015 RMS; real user
# speech is 10x higher. Tunable via env so you can dial it in for your room.
BARGE_IN_RMS_GATE = float(os.environ.get("BARGE_IN_RMS_GATE", "0.05"))
# require this many consecutive 32 ms frames above the gate before we fire
# barge-in. Single-frame echo bursts from AEC residual won't sustain; real
# user speech will. 4 frames ≈ 128 ms — balance between catching real
# interruptions (which AEC often partially attenuates during double-talk)
# and ignoring brief noise blips.
BARGE_IN_SUSTAIN_FRAMES = int(os.environ.get("BARGE_IN_SUSTAIN_FRAMES", "4"))

# Shortest audio we will hand to STT. Below roughly this, a clip carries less
# than a word and small ASR models return confident stock phrases instead of
# nothing. Kept low enough that a genuine one-word "yes"/"no" still gets through.
MIN_UTTERANCE_MS = int(os.environ.get("MIN_UTTERANCE_MS", "250"))
MIN_UTTERANCE_SAMPLES = SR * MIN_UTTERANCE_MS // 1000

HISTORY_FILE = Path(__file__).resolve().parent / ".voice_history.json"
HISTORY_MAX_TURNS = int(os.environ.get("HISTORY_MAX_TURNS", "8"))

SYSTEM_PROMPT = (
    "You are a friend on a voice call. Reply in 1 short sentence, sometimes 2. "
    "Talk like a person, not a chatbot — use contractions, casual phrasing, and "
    "the kind of fillers a human uses ('hmm', 'yeah', 'oh'). "
    "Never use bullet points, lists, code, or markdown. "
    "Never say 'as an AI' or add safety disclaimers. No em-dashes. "
    "If something is complex, give the gist in one breath and offer to dig in."
)

SENT_END = re.compile(r"[.!?](?:\s|$)|\n")


# ─────────────────────────────── helpers ──────────────────────────────


def f32_to_i16(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16)


def i16_to_f32(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32) / 32768.0


def linear_resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    ratio = dst_sr / src_sr
    new_len = int(round(len(x) * ratio))
    if new_len == 0:
        return np.zeros(0, dtype=x.dtype)
    idx = np.linspace(0, len(x) - 1, new_len)
    i0 = idx.astype(np.int64)
    i1 = np.clip(i0 + 1, 0, len(x) - 1)
    frac = (idx - i0).astype(x.dtype)
    return (1.0 - frac) * x[i0] + frac * x[i1]


# ─────────────────────────────── LLM ──────────────────────────────────


def stream_llm(history: list[dict], client: httpx.Client, cancel: threading.Event):
    payload = {
        "model": LLM_MODEL,
        "messages": history,
        "stream": True,
        "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"temperature": 0.85, "top_p": 0.95, "num_predict": 220},
    }
    with client.stream("POST", OLLAMA_URL, json=payload, timeout=120.0) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if cancel.is_set():
                return
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            tok = d.get("message", {}).get("content", "")
            if tok:
                yield tok
            if d.get("done"):
                return


def sentence_stream(token_iter):
    buf = ""
    for tok in token_iter:
        buf += tok
        last = -1
        for m in SENT_END.finditer(buf):
            last = m.end()
        if last > 0 and len(buf[:last].strip()) >= 6:
            yield buf[:last].strip()
            buf = buf[last:]
    if buf.strip():
        yield buf.strip()


# ─────────────────────────────── STT ──────────────────────────────────


def transcribe_utterance(stt: SttEngine, audio_f32: np.ndarray) -> str:
    return stt.transcribe(audio_f32)


# ─────────────────────────────── memory ───────────────────────────────


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                # cap at HISTORY_MAX_TURNS pairs immediately so the very first
                # reply isn't penalized by an unbounded persisted file.
                if len(data) > HISTORY_MAX_TURNS * 2:
                    data = data[-HISTORY_MAX_TURNS * 2 :]
                return data
        except Exception:
            pass
    return []


def save_history(history: list[dict]) -> None:
    turns = [m for m in history if m.get("role") in ("user", "assistant")]
    HISTORY_FILE.write_text(json.dumps(turns, indent=2))


def trim_history(history: list[dict]) -> list[dict]:
    sys_msgs = [m for m in history if m["role"] == "system"]
    turns = [m for m in history if m["role"] in ("user", "assistant")]
    if len(turns) > HISTORY_MAX_TURNS * 2:
        turns = turns[-HISTORY_MAX_TURNS * 2 :]
    return sys_msgs + turns


# ─────────────────────────────── models ───────────────────────────────


@dataclass
class Models:
    vad: VADIterator
    stt: SttEngine
    tts: TtsEngine
    turn: TurnDetector
    apm: AudioProcessingModule | None
    client: httpx.Client


def load_models() -> Models:
    # The silence timeout is a flat tax on every turn: we wait this long after
    # you stop making noise before doing anything. With no semantic detector it
    # has to be generous (500 ms) or natural mid-sentence pauses get treated as
    # end-of-turn — and it still cuts people off, because silence duration says
    # nothing about whether a sentence is finished.
    #
    # With a semantic detector in the loop, that job moves to the model, so the
    # timeout only has to be long enough to catch a breath. 200 ms costs ~19 ms
    # of inference and saves ~300 ms per turn.
    turn_choice = os.environ.get("TURN_DETECTOR", "off").strip().lower()
    semantic_turn = turn_choice not in ("off", "none", "0")
    default_silence = "200" if semantic_turn else "500"
    min_silence_ms = int(os.environ.get("VAD_MIN_SILENCE_MS", default_silence))

    print(f"[1/5] Silero VAD (min_silence {min_silence_ms} ms)...", flush=True)
    vad = VADIterator(
        load_silero_vad(),
        threshold=float(os.environ.get("VAD_THRESHOLD", "0.6")),
        sampling_rate=SR,
        min_silence_duration_ms=min_silence_ms,
        speech_pad_ms=100,
    )

    # Print before building, not after: loading a model can take tens of
    # seconds (or download gigabytes on first run) and silence reads as a hang.
    # Print the engine's own name after building, never a hardcoded guess at
    # the default — that drifts the moment a default changes and silently
    # misreports which model is actually running.
    print(f"[2/5] STT: {os.environ.get('STT_ENGINE', '(default)')} loading...", flush=True)
    stt = build_stt()
    print(f"      -> {stt.name}", flush=True)

    print(f"[3/5] TTS: {os.environ.get('TTS_ENGINE', '(default)')} loading...", flush=True)
    tts = build_tts()
    print(f"      -> {tts.name} @ {tts.sample_rate} Hz", flush=True)

    print(f"[4/5] turn detector: {os.environ.get('TURN_DETECTOR', 'off')}", flush=True)
    turn = build_turn_detector()

    apm = None
    if not HALF_DUPLEX_ONLY:
        print(f"[5/5] WebRTC AEC (stream delay {STREAM_DELAY_MS} ms)...", flush=True)
        # AEC + HPF only. Reasoning:
        # - AEC removes the agent's voice from the mic.
        # - NS, when on, was damaging real user speech (Moonshine garbles its
        #   output) for marginal residual reduction. The energy gate below
        #   handles residual better.
        # - AGC ducks user voice during loud TTS, killing barge-in. Stays off.
        # - HPF removes low-freq rumble, helps STT, no downside.
        apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=False,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        apm.set_stream_delay_ms(STREAM_DELAY_MS)
    else:
        print("[4/4] AEC disabled (HALF_DUPLEX=1)", flush=True)

    client = httpx.Client(timeout=httpx.Timeout(120.0))
    return Models(vad=vad, stt=stt, tts=tts, turn=turn, apm=apm, client=client)


def warmup(models: Models) -> None:
    print("warming up...", flush=True)
    t0 = time.perf_counter()

    models.stt.warmup()

    try:
        models.client.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "keep_alive": KEEP_ALIVE,
                "options": {"num_predict": 1},
            },
            timeout=120.0,
        )
    except Exception as e:
        print(f"  (LLM warmup: {e})")

    models.tts.warmup()

    print(f"  warmed in {(time.perf_counter()-t0)*1000:.0f}ms", flush=True)


# ─────────────────────────────── playback ─────────────────────────────


class TTSPlayer:
    """Streams TTS through the APM reverse stream, then to speakers, in 10 ms frames.

    Why play in 10 ms frames at 16 kHz: the APM requires its reverse stream
    to be exactly 10 ms wide, and AEC works best when the same audio that we
    feed to APM is what the speaker actually emits. So we resample Kokoro's
    24 kHz output down to 16 kHz once, then write it both ways in lockstep.
    """

    def __init__(self, tts: TtsEngine, apm: AudioProcessingModule | None):
        self.tts = tts
        self.apm = apm
        self.in_sr = tts.sample_rate  # 24000 for Kokoro and the mlx-audio models
        self.out_sr = SR  # 16000
        self.stream = sd.OutputStream(
            samplerate=self.out_sr, channels=1, dtype="int16", blocksize=APM_FRAME
        )
        self.stream.start()
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    def is_active(self) -> bool:
        with self._lock:
            return not self._cancel.is_set()

    def speak_sentence(self, sentence: str) -> None:
        with self._lock:
            self._cancel.clear()
            cancel = self._cancel

        residual = np.zeros(0, dtype=np.float32)
        for arr in self.tts.stream(sentence):
            if cancel.is_set():
                return
            if self.in_sr != self.out_sr:
                arr = linear_resample(arr, self.in_sr, self.out_sr).astype(np.float32)
            buf = np.concatenate([residual, arr])

            n_full = len(buf) // APM_FRAME
            for i in range(n_full):
                if cancel.is_set():
                    return
                frame_f32 = buf[i * APM_FRAME : (i + 1) * APM_FRAME]
                frame_i16 = f32_to_i16(frame_f32)
                if self.apm is not None:
                    af = AudioFrame(
                        data=frame_i16.tobytes(),
                        sample_rate=self.out_sr,
                        num_channels=1,
                        samples_per_channel=APM_FRAME,
                    )
                    self.apm.process_reverse_stream(af)
                self.stream.write(frame_i16)
            residual = buf[n_full * APM_FRAME :]

        if len(residual) > 0 and not cancel.is_set():
            pad = np.zeros(APM_FRAME - len(residual), dtype=np.float32)
            frame_i16 = f32_to_i16(np.concatenate([residual, pad]))
            if self.apm is not None:
                af = AudioFrame(
                    data=frame_i16.tobytes(),
                    sample_rate=self.out_sr,
                    num_channels=1,
                    samples_per_channel=APM_FRAME,
                )
                self.apm.process_reverse_stream(af)
            self.stream.write(frame_i16)

    def stop(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._cancel.set()
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass


# ─────────────────────────────── main loop ────────────────────────────


def main() -> int:
    models = load_models()
    warmup(models)

    persisted = load_history()
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}] + persisted
    if persisted:
        print(f"[memory] loaded {len(persisted)//2} prior turns")

    player = TTSPlayer(models.tts, models.apm)

    # mic queue holds 10 ms (APM_FRAME) chunks of float32 mono
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def mic_cb(indata, frames, t, status):  # noqa: ANN001
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy().astype(np.float32))

    in_stream = sd.InputStream(
        samplerate=SR,
        channels=1,
        dtype="float32",
        blocksize=APM_FRAME,  # 10 ms — matches APM
        callback=mic_cb,
    )
    in_stream.start()

    mode = "FULL-DUPLEX + AEC + barge-in" if models.apm is not None else "HALF-DUPLEX"
    print(f"\n=== ready ({mode}) — Ctrl+C to quit ===\n", flush=True)

    vad_buf = np.zeros(0, dtype=np.float32)  # accumulates AEC'd audio for VAD
    speech_buf: list[np.ndarray] = []
    active = False
    state = {
        "speaking": False,
        "tts_audible": False,
        "resume_at": 0.0,
        "user_just_ended": 0.0,
        "loud_streak": 0,         # consecutive AEC'd frames above the gate
        "barge_fired": False,     # disarms further barge-ins until next reply
    }
    llm_cancel = threading.Event()

    # cooldown after user finishes speaking, before barge-in can fire on the
    # agent's pre-audio phase (LLM still generating, no audio out yet). Without
    # this, the user's natural breathing or trailing breath after end-of-utterance
    # trips VAD and instantly cancels the agent's reply before it can speak.
    POST_USER_BARGE_LOCKOUT_S = 0.6

    # One long-lived worker handles every turn, rather than a thread per turn.
    # MLX keeps thread-local Metal state, and tearing that down when a thread
    # exits crashes the interpreter outright:
    #   Fatal Python error: PyThreadState_Get: ... the GIL is released
    # A thread per turn therefore died on the second turn once the STT slot
    # held an MLX model (Kokoro alone had tolerated it). Reproduced both ways:
    # per-turn threads crash at turn 1->2, a persistent worker ran 8/8 clean.
    turn_q: queue.Queue = queue.Queue()

    def turn_worker():
        while True:  # never returns — exiting is precisely what crashes
            audio_buf, cancel = turn_q.get()
            try:
                respond(audio_buf, models, history, player, cancel, state)
            except Exception as e:  # one bad turn must not kill the worker
                print(f"\n[turn failed] {type(e).__name__}: {e}", flush=True)
            finally:
                state["resume_at"] = time.monotonic() + TTS_TAIL_GRACE_MS / 1000.0
                state["speaking"] = False
                state["tts_audible"] = False
                models.vad.reset_states()

    threading.Thread(target=turn_worker, daemon=True).start()

    try:
        while True:
            mic_chunk_f32 = audio_q.get()  # 10 ms float32

            # In half-duplex fallback, drop mic frames while speaking.
            if models.apm is None and (
                state["speaking"] or time.monotonic() < state["resume_at"]
            ):
                continue

            # AEC: process mic through APM (subtracts speaker echo using reverse stream).
            if models.apm is not None:
                mic_i16 = f32_to_i16(mic_chunk_f32)
                af = AudioFrame(
                    data=mic_i16.tobytes(),
                    sample_rate=SR,
                    num_channels=1,
                    samples_per_channel=APM_FRAME,
                )
                models.apm.process_stream(af)
                cleaned_i16 = np.frombuffer(af.data, dtype=np.int16)
                cleaned = i16_to_f32(cleaned_i16)
            else:
                cleaned = mic_chunk_f32

            # Accumulate cleaned audio into 512-sample VAD frames.
            vad_buf = np.concatenate([vad_buf, cleaned])
            while len(vad_buf) >= VAD_FRAME:
                v_chunk = vad_buf[:VAD_FRAME]
                vad_buf = vad_buf[VAD_FRAME:]

                # Track sustained loud-frame streaks for barge-in. We compare
                # RMS on every VAD frame (32 ms). A real user voice mid-reply
                # produces a long streak; AEC residual produces brief spikes.
                rms = float(np.sqrt(np.mean(v_chunk**2)))
                if rms >= BARGE_IN_RMS_GATE:
                    state["loud_streak"] += 1
                else:
                    state["loud_streak"] = 0

                # Mid-TTS barge-in: if user has been loud for N consecutive
                # frames AND VAD currently agrees there's speech, cut the reply.
                # `barge_fired` disarms repeats; we re-arm when the next reply starts.
                if (
                    state["speaking"]
                    and state["tts_audible"]
                    and not state["barge_fired"]
                    and state["loud_streak"] >= BARGE_IN_SUSTAIN_FRAMES
                    and time.monotonic() - state["user_just_ended"] >= POST_USER_BARGE_LOCKOUT_S
                ):
                    print(
                        f"[barge-in] cutting reply (rms={rms:.3f} streak={state['loud_streak']})",
                        flush=True,
                    )
                    llm_cancel.set()
                    player.stop()
                    state["loud_streak"] = 0
                    state["barge_fired"] = True

                event = models.vad(torch.from_numpy(v_chunk), return_seconds=False)
                if event:
                    if "start" in event:
                        if not state["speaking"]:
                            if active:
                                # Already mid-utterance: the turn detector told
                                # us the thought wasn't finished, so this is the
                                # user resuming after a pause. Keep the buffer —
                                # resetting it here drops everything said before
                                # the pause and leaves STT decoding a fragment
                                # too short to get right.
                                speech_buf.append(v_chunk)
                            else:
                                speech_buf = [v_chunk]
                                print("🎙  listening...", flush=True)
                            active = True
                    elif "end" in event and active:
                        speech_buf.append(v_chunk)
                        audio = np.concatenate(speech_buf)

                        # Sub-word fragments make small STT models hallucinate
                        # fluent nonsense rather than return nothing — "No.",
                        # "Thank you." and similar stock phrases. Cheaper to
                        # refuse to transcribe them than to filter them after.
                        too_short = len(audio) < MIN_UTTERANCE_SAMPLES

                        # Silero says the sound stopped; the turn detector says
                        # whether the *thought* finished. If not, stay active so
                        # the continuation joins this same utterance instead of
                        # becoming a second, truncated one.
                        if too_short or not models.turn.is_complete(audio):
                            models.vad.reset_states()
                            continue

                        active = False
                        state["speaking"] = True
                        state["tts_audible"] = False
                        state["user_just_ended"] = time.monotonic()
                        state["loud_streak"] = 0
                        state["barge_fired"] = False  # re-arm for the new reply
                        llm_cancel = threading.Event()
                        turn_q.put((audio, llm_cancel))
                        models.vad.reset_states()
                elif active:
                    speech_buf.append(v_chunk)

    except KeyboardInterrupt:
        print("\nbye 👋")
    finally:
        save_history(history)
        in_stream.stop()
        in_stream.close()
        player.close()
        models.stt.close()
        models.tts.close()
        models.turn.close()
    return 0


def respond(
    audio: np.ndarray,
    models: Models,
    history: list[dict],
    player: TTSPlayer,
    cancel: threading.Event,
    state: dict,
) -> None:
    t0 = time.perf_counter()
    user_text = transcribe_utterance(models.stt, audio)
    t_stt = time.perf_counter() - t0
    if not user_text:
        print("(no speech detected)", flush=True)
        return
    print(f"You: {user_text}   [stt {t_stt*1000:.0f}ms]", flush=True)

    history.append({"role": "user", "content": user_text})
    history[:] = trim_history(history)

    print("AI: ", end="", flush=True)
    full_reply: list[str] = []
    first_audio_at: float | None = None
    t1 = time.perf_counter()

    try:
        for sent in sentence_stream(stream_llm(history, models.client, cancel)):
            if cancel.is_set():
                break
            print(sent + " ", end="", flush=True)
            full_reply.append(sent)
            if first_audio_at is None:
                first_audio_at = time.perf_counter() - t1
            state["tts_audible"] = True   # audio is now hitting the speaker; barge-in legit
            player.speak_sentence(sent)
            if cancel.is_set():
                break
    except Exception as e:
        print(f"\n[error] {e}", flush=True)

    print()
    if first_audio_at is not None:
        print(f"[ttfa {first_audio_at*1000:.0f}ms]", flush=True)

    if full_reply:
        history.append({"role": "assistant", "content": " ".join(full_reply)})
        save_history(history)


if __name__ == "__main__":
    sys.exit(main())
