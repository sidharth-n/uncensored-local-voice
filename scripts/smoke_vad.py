"""Smoke-test Silero VAD with a synthetic silence-speech-silence track.

Generates ~4s wav: 1s silence, 2s of macOS-say speech, 1s silence.
Feeds it through VADIterator in 512-sample chunks (the way the live agent does).
Reports detected start/end events.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from silero_vad import VADIterator, load_silero_vad

SR = 16000
VAD_FRAME = 512


def synth(text: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "u.aiff"
        subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
        data, sr = sf.read(str(aiff))
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SR:
            ratio = SR / sr
            new_len = int(len(data) * ratio)
            idx = (np.arange(new_len) / ratio).astype(np.int64)
            data = data[idx]
        return data.astype(np.float32)


def main() -> int:
    speech = synth("Hello there, how are you doing today?")
    silence_pre = np.zeros(SR, dtype=np.float32)
    silence_post = np.zeros(SR, dtype=np.float32)
    audio = np.concatenate([silence_pre, speech, silence_post])
    print(f"track: {len(audio)/SR:.2f}s  speech: {len(speech)/SR:.2f}s")

    vad = VADIterator(
        load_silero_vad(),
        threshold=0.5,
        sampling_rate=SR,
        min_silence_duration_ms=500,
        speech_pad_ms=100,
    )

    events = []
    t0 = time.perf_counter()
    for i in range(0, len(audio), VAD_FRAME):
        chunk = audio[i : i + VAD_FRAME]
        if len(chunk) < VAD_FRAME:
            chunk = np.pad(chunk, (0, VAD_FRAME - len(chunk)))
        ev = vad(torch.from_numpy(chunk), return_seconds=True)
        if ev:
            events.append((i / SR, ev))
            print(f"  t={i/SR:.2f}s -> {ev}")
    dt = time.perf_counter() - t0
    print(f"vad processed {len(audio)/SR:.2f}s of audio in {dt*1000:.0f}ms")
    print(f"events: {len(events)}")
    if not events:
        print("FAIL: no VAD events"); return 1
    starts = [e for _, e in events if "start" in e]
    ends = [e for _, e in events if "end" in e]
    print(f"starts: {len(starts)}  ends: {len(ends)}")
    if not starts or not ends:
        print("FAIL: missing start or end event"); return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
