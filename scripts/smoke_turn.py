"""Accuracy and latency check for the semantic turn detector.

The claim being tested: the detector can tell a finished thought from an
unfinished one, so the pipeline can stop relying on a fixed silence timeout.

Vendor numbers (92.63% overall, <60 ms CPU) come from cloud x64 hardware and
a 31k-sample test set. This is the same question asked of *this* machine with
*these* utterances, because a detector that is accurate but slow, or fast but
wrong about trailing conjunctions, is not usable here either way.

    uv run python scripts/smoke_turn.py
    TURN_DETECTOR=smartturn uv run python scripts/smoke_turn.py
"""

from __future__ import annotations

import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines import build_turn_detector  # noqa: E402

# Ground truth. The incomplete cases are the ones that matter: each ends on a
# word that a human would never stop on, which is precisely where a
# silence-timeout endpointer cuts you off mid-thought.
CASES: list[tuple[str, bool]] = [
    ("What should I make for dinner tonight?", True),
    ("I finished the report and sent it over.", True),
    ("Okay, that sounds good to me.", True),
    ("Tell me a joke.", True),
    ("What should I make for", False),
    ("I was thinking that maybe we could", False),
    ("The thing is, I really want to", False),
    ("Can you help me with the", False),
]


def synth(text: str, path: Path) -> np.ndarray:
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-o", str(aiff), text], check=True)
    data, sr = sf.read(str(aiff))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        ratio = 16000 / sr
        idx = (np.arange(int(len(data) * ratio)) / ratio).astype(np.int64)
        data = data[idx]
    aiff.unlink(missing_ok=True)
    return data.astype(np.float32)


def main() -> int:
    engine = os.environ.get("TURN_DETECTOR", "off")
    if engine in ("off", "none", "0"):
        print("TURN_DETECTOR is 'off' — nothing to test.")
        print("Re-run with: TURN_DETECTOR=smartturn uv run python scripts/smoke_turn.py")
        return 0

    t0 = time.perf_counter()
    det = build_turn_detector()
    print(f"=== {det.name} (load {(time.perf_counter()-t0)*1000:.0f} ms) ===\n")

    latencies: list[float] = []
    correct = 0
    with tempfile.TemporaryDirectory() as td:
        # One untimed call first: ORT allocates arenas and transformers builds
        # its mel filterbank on the first pass, which would skew the median.
        det.is_complete(synth("Warming up.", Path(td) / "w.wav"))

        for i, (text, expected) in enumerate(CASES):
            audio = synth(text, Path(td) / f"c{i}.wav")
            t0 = time.perf_counter()
            got = det.is_complete(audio)
            dt = (time.perf_counter() - t0) * 1000
            latencies.append(dt)
            ok = got == expected
            correct += ok
            prob = getattr(det, "last_probability", None)
            pstr = f"p={prob:.3f}" if prob is not None else "p=?"
            print(
                f"  {'PASS' if ok else 'FAIL'}  {pstr}  "
                f"want={'complete' if expected else 'incomplete':<10} "
                f"got={'complete' if got else 'incomplete':<10} "
                f"{dt:6.1f} ms   {text!r}"
            )

    det.close()
    print(f"\n  accuracy : {correct}/{len(CASES)}")
    print(f"  latency  : {statistics.median(latencies):.1f} ms median, "
          f"{min(latencies):.1f}–{max(latencies):.1f} ms")
    print("\n  This runs once per end-of-speech event, not per 32 ms frame —")
    print("  so it is additive to turn latency, and only worth it if it lets")
    print("  min_silence_duration_ms drop by more than it costs.")
    return 0 if correct == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
