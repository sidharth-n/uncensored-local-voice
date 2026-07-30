"""Multi-turn regression test — no mic, no speaking required.

Exists because of a crash that every other test in this repo was structurally
incapable of catching. smoke_pipeline.py runs exactly one turn on the main
thread; the live agent runs many turns on a worker thread. MLX keeps
thread-local Metal state, and destroying it when a thread exits kills the
interpreter:

    Fatal Python error: PyThreadState_Get: the function must be called with
    the GIL held ... but the GIL is released

So the agent used to spawn a thread per turn and die on the second one, while
every scripted test passed. This drives the real engines over several turns on
a single persistent worker — the same shape as voice_agent's turn_worker — and
fails loudly if the process cannot survive it.

    uv run python scripts/smoke_multiturn.py
    uv run python scripts/smoke_multiturn.py --turns 12
    STT_ENGINE=moonshine uv run python scripts/smoke_multiturn.py
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engines import build_stt, build_tts, build_turn_detector  # noqa: E402

SR = 16000
VOICE_WAV = ROOT / "my_voice.wav"


def probe_audio(seconds: float = 4.0) -> np.ndarray:
    """Prefer a real recording; fall back to macOS `say`, then to noise."""
    if VOICE_WAV.exists():
        a, sr = sf.read(str(VOICE_WAV))
        a = (a.mean(axis=1) if a.ndim > 1 else a).astype(np.float32)
        if sr != SR:
            ratio = SR / sr
            a = a[(np.arange(int(len(a) * ratio)) / ratio).astype(np.int64)]
        return a[: int(seconds * SR)]

    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        aiff = Path(td) / "p.aiff"
        try:
            subprocess.run(
                ["say", "-v", "Samantha", "-o", str(aiff),
                 "What is the weather like tomorrow?"], check=True)
            d, sr = sf.read(str(aiff))
            d = (d.mean(axis=1) if d.ndim > 1 else d).astype(np.float32)
            ratio = SR / sr
            return d[(np.arange(int(len(d) * ratio)) / ratio).astype(np.int64)]
        except Exception:
            # Last resort: the point of this test is thread lifetime, not
            # transcription quality, so noise still exercises the crash path.
            return np.random.randn(int(seconds * SR)).astype(np.float32) * 0.05


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=8)
    args = ap.parse_args()

    audio = probe_audio()
    print(f"=== multi-turn regression: {args.turns} turns, "
          f"{len(audio)/SR:.1f}s probe audio")

    stt, tts, turn = build_stt(), build_tts(), build_turn_detector()
    print(f"    stt={stt.name}  tts={tts.name}  turn={turn.name}\n")
    stt.warmup()
    tts.warmup()

    q: queue.Queue = queue.Queue()
    done = threading.Event()
    state = {"n": 0, "err": None}

    def worker() -> None:
        while True:  # mirrors voice_agent.turn_worker: never exits
            i = q.get()
            try:
                turn.is_complete(audio)
                text = stt.transcribe(audio)
                samples = sum(c.shape[0] for c in tts.stream("Reply number %d." % i))
                state["n"] += 1
                print(f"  turn {i + 1}/{args.turns} ok  "
                      f"stt={text[:36]!r} tts={samples} samples", flush=True)
            except Exception as e:  # noqa: BLE001
                state["err"] = f"{type(e).__name__}: {e}"
                done.set()
                return
            if state["n"] >= args.turns:
                done.set()

    threading.Thread(target=worker, daemon=True).start()
    for i in range(args.turns):
        q.put(i)

    if not done.wait(timeout=60 + 20 * args.turns):
        print("\nFAIL: timed out — worker stalled")
        return 1
    time.sleep(0.2)

    if state["err"]:
        print(f"\nFAIL: {state['err']}")
        return 1
    if state["n"] < args.turns:
        print(f"\nFAIL: only {state['n']}/{args.turns} turns completed")
        return 1

    print(f"\nPASS: {state['n']}/{args.turns} turns, process alive, worker alive")
    print("      (a fatal GIL error printed after this line would mean the")
    print("       crash moved to interpreter shutdown, which is cosmetic)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
