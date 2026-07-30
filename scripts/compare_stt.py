"""Record your real voice once, then transcribe it with every STT engine.

Why this exists: the agent was mishearing badly, and there were two plausible
causes that no amount of reading could separate — the model being weak on this
speaker's accent, or the pipeline mangling the audio before the model sees it
(AEC damage, VAD fragmenting an utterance into pieces too short to decode).

Synthetic `say` audio cannot answer that. Only a real clip of the actual
speaker can, so this records one and reuses the same samples for every engine,
which also makes the comparison fair.

    # record 8 s and compare the default set
    uv run python scripts/compare_stt.py

    # reuse a clip you already recorded
    uv run python scripts/compare_stt.py --wav myvoice.wav

    # test the fragmenting theory: transcribe short slices of the same clip
    uv run python scripts/compare_stt.py --wav myvoice.wav --fragments
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines import STT_MODELS, MlxAudioStt, MoonshineStt  # noqa: E402

SR = 16000
DEFAULT_WAV = ROOT / "my_voice.wav"


def record(seconds: float, path: Path) -> np.ndarray:
    print(f"\n  Recording {seconds:.0f}s — speak now, normally, as you would to the agent.")
    print("  Suggested: \"What's the weather going to be like tomorrow evening?\"\n")
    for i in (3, 2, 1):
        print(f"    {i}...", flush=True)
        time.sleep(1)
    print("    GO", flush=True)
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    audio = audio[:, 0]
    sf.write(str(path), audio, SR)
    peak = float(np.max(np.abs(audio)))
    print(f"    done — {len(audio)/SR:.1f}s, peak {peak:.3f}")
    if peak < 0.02:
        print("    WARNING: that is almost silent. Check the input device.")
    return audio


def run(label: str, make, audio: np.ndarray) -> None:
    try:
        t0 = time.perf_counter()
        eng = make()
        load_ms = (time.perf_counter() - t0) * 1000
        eng.warmup()
        t0 = time.perf_counter()
        text = eng.transcribe(audio)
        dt = (time.perf_counter() - t0) * 1000
        eng.close()
        print(f"\n  {label}")
        print(f"    {dt:7.0f} ms   (load {load_ms:.0f} ms)")
        print(f"    -> {text!r}")
    except Exception as e:
        print(f"\n  {label}")
        print(f"    FAILED: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=None, help="use an existing 16 kHz mono wav")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--engines", default="moonshine,parakeet,nemotron")
    ap.add_argument(
        "--fragments",
        action="store_true",
        help="also transcribe 1s/2s/3s slices, to show how short audio degrades",
    )
    args = ap.parse_args()

    if args.wav:
        audio, sr = sf.read(args.wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != SR:
            ratio = SR / sr
            idx = (np.arange(int(len(audio) * ratio)) / ratio).astype(np.int64)
            audio = audio[idx]
        print(f"=== using {args.wav} ({len(audio)/SR:.1f}s)")
    else:
        audio = record(args.seconds, DEFAULT_WAV)
        print(f"=== saved to {DEFAULT_WAV.name} — reuse it with --wav {DEFAULT_WAV.name}")

    print("\n" + "=" * 68)
    print("FULL UTTERANCE — same samples through every engine")
    print("=" * 68)

    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name == "moonshine":
            run("moonshine (current default, English-only)",
                lambda: MoonshineStt(), audio)
        elif name in STT_MODELS:
            run(f"{name}  ({STT_MODELS[name]})",
                lambda n=name: MlxAudioStt(STT_MODELS[n],
                                           language=os.environ.get("STT_LANGUAGE")),
                audio)
        else:
            run(name, lambda n=name: MlxAudioStt(n), audio)

    if args.fragments:
        # The live agent was splitting speech into pieces and decoding each
        # alone. If short slices produce confident nonsense ("No.", "Thank
        # you."), the fragmenting is the bug, not the model.
        print("\n" + "=" * 68)
        print("FRAGMENTS — how each engine behaves on truncated audio")
        print("=" * 68)
        eng = MoonshineStt()
        eng.warmup()
        for secs in (0.5, 1.0, 2.0, 3.0):
            n = int(secs * SR)
            if n >= len(audio):
                break
            print(f"\n  first {secs}s -> {eng.transcribe(audio[:n])!r}")
        eng.close()

    print("\n  Judge by which transcript matches what you actually said.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
