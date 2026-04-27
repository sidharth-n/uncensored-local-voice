"""End-to-end smoke test that exercises STT -> LLM -> TTS without a live mic.

Generates an utterance with macOS `say`, transcribes it with Moonshine,
sends to Ollama (SuperGemma4, think=false) with streaming, and synthesizes
the reply with Kokoro. Reports latency at each stage.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent import (  # noqa: E402
    LLM_MODEL,
    OLLAMA_URL,
    SYSTEM_PROMPT,
    sentence_stream,
    stream_llm,
    transcribe_utterance,
)
from kokoro_mlx import DEFAULT_VOICE, KokoroTTS  # noqa: E402
from moonshine_voice import get_model_for_language  # noqa: E402
from moonshine_voice.moonshine_api import ModelArch  # noqa: E402
from moonshine_voice.transcriber import Transcriber  # noqa: E402


def synth_wav(text: str, out_path: Path) -> None:
    """Use macOS `say` to write a wav, then resample to 16 kHz mono float32."""
    aiff = out_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    data, sr = sf.read(str(aiff))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        # naive linear resample to 16 kHz
        from math import floor

        ratio = 16000 / sr
        new_len = floor(len(data) * ratio)
        idx = (np.arange(new_len) / ratio).astype(np.int64)
        data = data[idx]
        sr = 16000
    sf.write(str(out_path), data.astype(np.float32), sr, subtype="FLOAT")
    aiff.unlink(missing_ok=True)


def main() -> int:
    test_utterance = "Hey, what should I make for dinner tonight?"
    print(f"=== test utterance: {test_utterance!r}")

    # 1. Synthesize test audio
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "test.wav"
        synth_wav(test_utterance, wav)
        audio, sr = sf.read(str(wav))
        audio = audio.astype(np.float32)
        print(f"[audio] {audio.shape[0]} samples @ {sr}Hz, {audio.shape[0]/sr:.2f}s")

        # 2. STT
        print("\n--- STT (Moonshine) ---")
        t0 = time.perf_counter()
        stt_path, stt_arch = get_model_for_language("en", ModelArch.BASE)
        stt = Transcriber(model_path=str(stt_path), model_arch=stt_arch)
        stt.start()
        load_dt = time.perf_counter() - t0
        print(f"  load: {load_dt*1000:.0f}ms")
        t0 = time.perf_counter()
        text = transcribe_utterance(stt, audio)
        stt_dt = time.perf_counter() - t0
        stt.stop(); stt.close()
        print(f"  transcribe: {stt_dt*1000:.0f}ms")
        print(f"  result: {text!r}")
        if not text.strip():
            print("FAIL: empty transcription"); return 1

    # 3. LLM stream + sentence split
    print("\n--- LLM (SuperGemma4, think=False, stream=True) ---")
    history = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    client = httpx.Client(timeout=httpx.Timeout(120.0))
    t0 = time.perf_counter()
    sentences: list[str] = []
    ttft = None
    for sent in sentence_stream(stream_llm(history, client)):
        if ttft is None:
            ttft = time.perf_counter() - t0
            print(f"  TTFT (first sentence): {ttft*1000:.0f}ms")
        print(f"  sent: {sent!r}")
        sentences.append(sent)
    full = " ".join(sentences)
    print(f"  total: {len(full)} chars, {len(sentences)} sentences")
    if not sentences:
        print("FAIL: no LLM output"); return 1

    # 4. TTS
    print("\n--- TTS (Kokoro MLX) ---")
    t0 = time.perf_counter()
    tts = KokoroTTS.from_pretrained()
    print(f"  load: {(time.perf_counter()-t0)*1000:.0f}ms")
    audio_chunks = []
    t0 = time.perf_counter()
    first = None
    total_samples = 0
    for sent in sentences:
        for chunk in tts.generate_stream(sent, voice=DEFAULT_VOICE):
            if first is None:
                first = time.perf_counter() - t0
            arr = np.asarray(chunk, dtype=np.float32).flatten()
            audio_chunks.append(arr)
            total_samples += arr.shape[0]
    total_dt = time.perf_counter() - t0
    audio_seconds = total_samples / tts.SAMPLE_RATE
    rt_factor = audio_seconds / total_dt
    print(f"  TTFA: {first*1000:.0f}ms")
    print(f"  total synth: {total_dt*1000:.0f}ms for {audio_seconds:.2f}s audio ({rt_factor:.1f}x realtime)")

    # Save final audio so user can listen
    out_wav = ROOT / "out_reply.wav"
    sf.write(str(out_wav), np.concatenate(audio_chunks), tts.SAMPLE_RATE)
    print(f"\nWrote reply audio to {out_wav}")
    print(f"\nReply text: {full}")

    # Latency summary
    print("\n=== summary ===")
    print(f"STT      : {stt_dt*1000:.0f}ms")
    print(f"LLM TTFT : {ttft*1000:.0f}ms")
    print(f"TTS TTFA : {first*1000:.0f}ms")
    end_to_end_until_audio = stt_dt + ttft + first
    print(f"end-to-end (stt+llm-ttft+tts-ttfa): {end_to_end_until_audio*1000:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
