"""Swappable engine slots for the voice pipeline.

Why this exists: every component in the pipeline (STT, TTS, end-of-turn) is a
model choice we want to A/B on real hardware, not a permanent decision. Each
slot is a small protocol plus one adapter per backend, selected by an env var.

The defaults reproduce the original hard-wired pipeline exactly — Moonshine STT,
Kokoro TTS, no semantic turn detection — so an unset environment behaves
identically to before this module existed.

    STT_ENGINE=moonshine   (default)
    TTS_ENGINE=kokoro      (default)
    TURN_DETECTOR=off      (default)
"""

from __future__ import annotations

import os
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

SR = 16000  # pipeline-wide sample rate; TTS adapters resample to this upstream


# ─────────────────────────────── protocols ────────────────────────────


@runtime_checkable
class SttEngine(Protocol):
    """Utterance-at-a-time speech to text.

    The pipeline buffers a whole utterance behind VAD before transcribing, so
    engines only need a batch call — not incremental partial hypotheses.
    """

    name: str

    def transcribe(self, audio_f32: np.ndarray) -> str: ...
    def warmup(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class TtsEngine(Protocol):
    """Sentence-at-a-time speech synthesis.

    `stream` yields float32 mono chunks at `sample_rate`. The player resamples
    to 16 kHz for both the speaker and the AEC reverse stream, so adapters
    report their native rate rather than resampling themselves.
    """

    name: str
    sample_rate: int

    def stream(self, sentence: str) -> Iterator[np.ndarray]: ...
    def warmup(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class TurnDetector(Protocol):
    """Decides whether a VAD-delimited utterance is actually finished.

    Silero tells us the user stopped making sound; it cannot tell us whether
    they finished their thought. A semantic detector answers the second
    question, which is what stops the agent cutting people off mid-sentence.
    """

    name: str

    def is_complete(self, audio_f32: np.ndarray) -> bool: ...
    def close(self) -> None: ...


# ─────────────────────────────── STT adapters ─────────────────────────


class MoonshineStt:
    """Moonshine via the `moonshine_voice` package. English only.

    MEDIUM_STREAMING is more accurate than BASE and STT sits well inside our
    latency budget, so we spend the extra compute on quality. The _STREAMING
    variant works fine for one-shot `transcribe_without_streaming` calls.
    """

    name = "moonshine"

    def __init__(self, language: str = "en", arch: str | None = None) -> None:
        from moonshine_voice import get_model_for_language
        from moonshine_voice.moonshine_api import ModelArch
        from moonshine_voice.transcriber import Transcriber

        resolved = getattr(ModelArch, arch) if arch else ModelArch.MEDIUM_STREAMING
        path, model_arch = get_model_for_language(language, resolved)
        self._t = Transcriber(model_path=str(path), model_arch=model_arch)
        self._t.start()

    def transcribe(self, audio_f32: np.ndarray) -> str:
        transcript = self._t.transcribe_without_streaming(
            audio_f32.tolist(), sample_rate=SR
        )
        parts = []
        for line in getattr(transcript, "lines", []) or []:
            text = getattr(line, "text", "") or ""
            if text.strip():
                parts.append(text.strip())
        return " ".join(parts).strip()

    def warmup(self) -> None:
        self.transcribe(np.zeros(SR, dtype=np.float32))

    def close(self) -> None:
        try:
            self._t.stop()
            self._t.close()
        except Exception:
            pass


class MlxAudioStt:
    """Any STT model in mlx-audio's registry, MLX-native on Apple Silicon.

    Gives us Parakeet TDT v3 (25 European languages), Nemotron 3.5 ASR
    (40 locales, language-ID prompting), Whisper, Canary and MMS behind one
    adapter — all substantially larger and better-trained than Moonshine,
    which is English-only and small enough to hallucinate on short clips.

    mlx-audio's generate() takes a file path, so we spill the utterance to a
    temp wav. At utterance rate that write is microseconds against hundreds of
    milliseconds of inference, and it keeps us on the library's supported path
    instead of reaching into its internals.
    """

    name = "mlxaudio"

    def __init__(self, model_id: str, language: str | None = None) -> None:
        from mlx_audio.stt import load

        self.model_id = model_id
        self.language = language
        self.name = f"mlxaudio:{model_id.split('/')[-1]}"
        self._model = load(model_id)

    def transcribe(self, audio_f32: np.ndarray) -> str:
        import tempfile
        import wave

        audio = np.asarray(audio_f32, dtype=np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:  # keep within [-1, 1] before int16 conversion clips it
            audio = audio / peak
        pcm = (audio * 32767.0).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            with wave.open(tmp.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(pcm.tobytes())
            kwargs = {"language": self.language} if self.language else {}
            try:
                result = self._model.generate(tmp.name, **kwargs)
            except TypeError:
                # not every model in the registry accepts a language kwarg
                result = self._model.generate(tmp.name)

        text = getattr(result, "text", None)
        if text is None:  # some models yield segments instead of a flat string
            segs = getattr(result, "segments", None) or []
            text = " ".join(getattr(s, "text", "") or "" for s in segs)
        return (text or "").strip()

    def warmup(self) -> None:
        self.transcribe(np.zeros(SR, dtype=np.float32))

    def close(self) -> None:
        self._model = None


# ─────────────────────────────── TTS adapters ─────────────────────────


class KokoroTts:
    """Kokoro 82M via `kokoro_mlx`. Fast, English-centric, 24 kHz out.

    Pinned to kokoro-mlx 0.1.1 by the interpreter: 0.1.2 requires Python <3.13
    and this venv is 3.13. That ceiling is one reason the TTS slot exists.
    """

    name = "kokoro"

    def __init__(self, voice: str | None = None) -> None:
        from kokoro_mlx import DEFAULT_VOICE, KokoroTTS

        self._tts = KokoroTTS.from_pretrained()
        self._voice = voice or DEFAULT_VOICE
        self.sample_rate = int(self._tts.SAMPLE_RATE)  # 24000

    def stream(self, sentence: str) -> Iterator[np.ndarray]:
        for chunk in self._tts.generate_stream(sentence, voice=self._voice):
            yield np.asarray(chunk, dtype=np.float32).flatten()

    def warmup(self) -> None:
        for _ in self.stream("Hi."):
            pass

    def close(self) -> None:
        pass


# ─────────────────────────── turn detectors ───────────────────────────


class NoTurnDetector:
    """The original behaviour: trust Silero's end-of-speech and reply.

    Kept as a named engine rather than a None check so the pipeline has one
    code path, and so `TURN_DETECTOR=off` stays a first-class, testable choice.
    """

    name = "off"

    def is_complete(self, audio_f32: np.ndarray) -> bool:
        return True

    def close(self) -> None:
        pass


class SmartTurnDetector:
    """Semantic end-of-turn detection via pipecat-ai/smart-turn v3.2 (ONNX).

    An 8M-param Whisper-Tiny backbone with a classifier head, run on CPU
    through onnxruntime. It answers "did this person finish their thought",
    which Silero cannot: Silero only knows sound stopped, so the pipeline had
    to wait out a fixed silence timeout and still cut people off on pauses.

    We deliberately depend on the raw .onnx weights plus transformers'
    WhisperFeatureExtractor rather than pulling in pipecat as a framework —
    the model is BSD-2-Clause and the preprocessing is ~10 lines.

    Cost: one inference per end-of-speech event, not per audio frame, so it
    sits outside the 32 ms VAD loop entirely.

    Note it covers 23 languages and Malayalam is not among them; worse, it
    reads lexical content rather than pure prosody, so it should not be
    assumed to generalize. `TURN_LANGUAGES` gates which languages use it.
    """

    name = "smartturn"

    HF_REPO = "pipecat-ai/smart-turn-v3"
    DEFAULT_FILE = "smart-turn-v3.2-cpu.onnx"
    WINDOW_S = 8  # the model is trained on a fixed 8 s window at 16 kHz

    def __init__(self, threshold: float = 0.5, filename: str | None = None) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        self.threshold = threshold
        path = hf_hub_download(
            repo_id=self.HF_REPO, filename=filename or self.DEFAULT_FILE
        )

        # Single-threaded sequential execution: this runs on the audio thread
        # between turns, and letting ORT spin up a thread pool per call costs
        # more than the inference itself at this model size.
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(path, sess_options=so)
        self._fx = WhisperFeatureExtractor(chunk_length=self.WINDOW_S)
        self.last_probability: float | None = None

    def _window(self, audio_f32: np.ndarray) -> np.ndarray:
        """Keep the last 8 s, or left-pad with silence to reach 8 s.

        Left-padding matters: the decision is about how the utterance *ended*,
        so the tail must stay anchored to the end of the window.
        """
        n = self.WINDOW_S * SR
        if len(audio_f32) > n:
            return audio_f32[-n:]
        if len(audio_f32) < n:
            return np.pad(audio_f32, (n - len(audio_f32), 0), mode="constant")
        return audio_f32

    def is_complete(self, audio_f32: np.ndarray) -> bool:
        audio = self._window(np.asarray(audio_f32, dtype=np.float32))
        inputs = self._fx(
            audio,
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=self.WINDOW_S * SR,
            truncation=True,
            do_normalize=True,
        )
        feats = np.expand_dims(inputs.input_features.squeeze(0).astype(np.float32), 0)
        prob = float(self._session.run(None, {"input_features": feats})[0][0].item())
        self.last_probability = prob
        return prob > self.threshold

    def close(self) -> None:
        self._session = None


# ─────────────────────────────── registry ─────────────────────────────


# Shorthands so callers say STT_ENGINE=parakeet rather than pasting a repo id.
STT_MODELS = {
    "parakeet": "mlx-community/parakeet-tdt-0.6b-v3",
    "nemotron": "mlx-community/nemotron-3.5-asr-streaming-0.6b",
    "nemotron-8bit": "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
    "whisper": "mlx-community/whisper-large-v3-turbo",
}


def build_stt() -> SttEngine:
    choice = os.environ.get("STT_ENGINE", "moonshine").strip().lower()
    if choice == "moonshine":
        return MoonshineStt(
            language=os.environ.get("STT_LANGUAGE", "en"),
            arch=os.environ.get("MOONSHINE_ARCH") or None,
        )
    if choice in STT_MODELS or "/" in choice:
        return MlxAudioStt(
            model_id=STT_MODELS.get(choice, os.environ.get("STT_ENGINE", "")),
            language=os.environ.get("STT_LANGUAGE") or None,
        )
    raise SystemExit(
        f"unknown STT_ENGINE={choice!r}. "
        f"available: moonshine, {', '.join(STT_MODELS)}, or an mlx-audio repo id"
    )


def build_tts() -> TtsEngine:
    choice = os.environ.get("TTS_ENGINE", "kokoro").strip().lower()
    if choice == "kokoro":
        return KokoroTts(voice=os.environ.get("TTS_VOICE") or None)
    raise SystemExit(
        f"unknown TTS_ENGINE={choice!r}. available: kokoro"
    )


def build_turn_detector() -> TurnDetector:
    choice = os.environ.get("TURN_DETECTOR", "off").strip().lower()
    if choice in ("off", "none", "0"):
        return NoTurnDetector()
    if choice in ("smartturn", "smart_turn", "smart-turn"):
        # Default 0.6 rather than the model's own 0.5. Two reasons, one solid
        # and one weak, so treat it as a starting point and not a tuned value:
        #
        #  - Solid: the costs are asymmetric. Declaring "complete" too eagerly
        #    talks over the user; declaring "incomplete" too eagerly just waits
        #    a beat longer. Requiring more confidence to interrupt is right.
        #  - Weak: on scripts/smoke_turn.py's 8 cases, 0.5 misses one trailing
        #    "...I really want to" at p=0.530 and 0.6 gets all eight. That is 8
        #    samples of synthetic speech, not evidence — the vendor calibrated
        #    0.5 on 31,527 real samples. Revisit against your own voice.
        return SmartTurnDetector(
            threshold=float(os.environ.get("TURN_THRESHOLD", "0.6")),
            filename=os.environ.get("SMART_TURN_FILE") or None,
        )
    raise SystemExit(
        f"unknown TURN_DETECTOR={choice!r}. available: off, smartturn"
    )
