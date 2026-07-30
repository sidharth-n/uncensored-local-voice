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

# A deliberately hostile passage. Every element here is something a voice
# assistant has to get right and small ASR models routinely fail: a time, two
# personal names, an Indian place name, a model name containing digits, two
# similar quantities, the third/thirteenth ordinal trap, and a currency amount.
HARD_PASSAGE = (
    "Remind me at five thirty tomorrow to email Anjali about the "
    "Thiruvananthapuram deployment. Check whether the Qwen three point six "
    "model needs sixteen gigabytes or twenty two, and whether the API quota "
    "resets on the third or the thirteenth. If it is under two hundred and "
    "fifty rupees, just buy it."
)


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


_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
# Ordinals map to their cardinal value so "third" and "3rd" score as equal.
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19,
    "twentieth": 20, "thirtieth": 30,
}


# Spelling out an abbreviation is a formatting choice, not a mishearing.
_ALIASES = {
    "gb": "gigabytes", "gigabyte": "gigabytes", "gigs": "gigabytes",
    "mb": "megabytes", "kb": "kilobytes", "tb": "terabytes",
    "rs": "rupees", "inr": "rupees", "rupee": "rupees",
    "a.p.i": "api", "okay": "ok",
}


def normalize(text: str) -> list[str]:
    """Reduce a transcript to comparable tokens.

    Formatting is not a recognition error. One engine writes "5:30", another
    "five thirty"; one writes "22", another "twenty two"; one writes "3rd",
    another "third". Scoring those as mistakes would rank engines on their
    output conventions instead of on what they heard. So number words are
    parsed to values, compound phrases combined ("twenty two" -> 22, "two
    hundred fifty" -> 250), ordinals folded to cardinals, and times split on
    the colon. Punctuation and case are dropped.
    """
    import re

    text = re.sub(r"[^\w\s:]", " ", text.lower())
    text = text.replace(":", " ")
    raw = [t for t in text.split() if t]

    # Digit forms first: "22" -> 22, "3rd" -> 3, "16gb" left as a word.
    toks: list[object] = []
    for t in raw:
        m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)?", t)
        toks.append(int(m.group(1)) if m else t)

    # Fold number-word phrases into single values.
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if isinstance(t, int):
            out.append(str(t))
            i += 1
            continue
        if t in _ORDINALS:
            out.append(str(_ORDINALS[t]))
            i += 1
            continue
        if t in _UNITS or t in _TENS:
            # English compounds run large-to-small: "twenty two" is 22, but
            # "five thirty" is not 35 — it is a time, two separate numbers.
            # So a tens word may absorb a following unit, never the reverse.
            # Absorb a word only if its slot in the current number is still
            # free: tens need (cur % 100 == 0), units need (cur % 10 == 0).
            # That makes "twenty two" -> 22 and "two hundred and fifty" -> 250,
            # while "five thirty" stops at 5 because thirty's slot is taken.
            total, cur = 0, 0
            while i < len(toks) and isinstance(toks[i], str):
                w = toks[i]
                if w in _TENS:
                    if cur % 100:
                        break
                    cur += _TENS[w]
                elif w in _UNITS:
                    v = _UNITS[w]
                    if v >= 10:  # teens fill the tens and units slots together
                        if cur % 100:
                            break
                    elif cur % 10:
                        break
                    cur += v
                elif w == "hundred":
                    cur = (cur or 1) * 100
                elif w == "thousand":
                    total += (cur or 1) * 1000
                    cur = 0
                elif w == "and" and cur:
                    pass  # "two hundred and fifty" is one number
                else:
                    break
                i += 1
            out.append(str(total + cur))
            continue
        out.append(_ALIASES.get(t, t))
        i += 1

    # "three point six" and "3.6" should match: drop a spoken decimal point
    # sitting between two numbers, since the digit form loses it to punctuation.
    cleaned: list[str] = []
    for j, tok in enumerate(out):
        if (
            tok == "point"
            and 0 < j < len(out) - 1
            and out[j - 1].isdigit()
            and out[j + 1].isdigit()
        ):
            continue
        cleaned.append(tok)
    return cleaned


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Word error rate by Levenshtein distance over normalized tokens."""
    r, h = normalize(reference), normalize(hypothesis)
    if not r:
        return 0.0, 0, 0
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    errs = int(d[len(r), len(h)])
    return errs / len(r), errs, len(r)


def run(label: str, make, audio: np.ndarray, reference: str | None = None) -> None:
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
        if reference:
            rate, errs, total = wer(reference, text)
            print(f"    WER {rate*100:5.1f}%  ({errs} errors / {total} words)"
                  f"   {dt:.0f} ms   (load {load_ms:.0f} ms)")
        else:
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
        "--reference",
        default=None,
        help="what you actually said; enables automatic WER scoring",
    )
    ap.add_argument(
        "--hard",
        action="store_true",
        help="print the hard passage to read, and score against it",
    )
    ap.add_argument(
        "--fragments",
        action="store_true",
        help="also transcribe 1s/2s/3s slices, to show how short audio degrades",
    )
    args = ap.parse_args()

    reference = args.reference
    if args.hard:
        reference = HARD_PASSAGE
        if args.seconds < 25:
            args.seconds = 25  # the passage needs ~22 s at a normal pace
        print("=" * 68)
        print("READ THIS ALOUD, at your normal speaking pace:")
        print("=" * 68)
        print(f"\n{HARD_PASSAGE}\n")
        print("=" * 68)

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
    if reference:
        print("scored against the reference; number/punctuation formatting is")
        print("normalized away, so WER reflects what was heard, not how it prints")
    print("=" * 68)

    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name == "moonshine":
            run("moonshine (English-only)", lambda: MoonshineStt(), audio, reference)
        elif name in STT_MODELS:
            run(f"{name}  ({STT_MODELS[name]})",
                lambda n=name: MlxAudioStt(STT_MODELS[n],
                                           language=os.environ.get("STT_LANGUAGE")),
                audio, reference)
        else:
            run(name, lambda n=name: MlxAudioStt(n), audio, reference)

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
