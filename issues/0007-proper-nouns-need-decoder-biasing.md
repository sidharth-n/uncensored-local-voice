---
id: 0007
title: Proper nouns are unrecognisable by every STT engine — needs biasing, not a bigger model
status: open
priority: moderate
area: stt
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Names, places and product names are consistently wrong, and **no model choice fixes
it** — all four engines tested failed the same words.

## Why it matters

"Remind me to email Anjali about the Thiruvananthapuram deployment" is exactly the kind
of thing this agent is for. Getting the verb right and the name wrong still produces a
useless result.

## Evidence

Hard-passage test, 2026-07-30, same recording through four engines:

| Engine | "Thiruvananthapuram" | "Qwen" |
|---|---|---|
| moonshine | Tiruvannamalai | quen |
| parakeet | Tirunanda Puram | Quen |
| nemotron | Srindapuram | Quen |
| whisper-large-v3-turbo | Tirunthapuram | QEN |

Four different wrong answers, zero correct. Note nemotron (8.9% overall WER, the most
accurate engine tested) failed it too — this is not a capability gap that scales away.

Whisper additionally invented a unit ("22GB" for "22") and heard "API cotta" for "API
quota".

## Approach

Decoder biasing / hotword boosting, not a larger model:

- `parakeet-mlx` exposes decoding config; check whether the TDT decoder supports
  external LM shallow fusion or a word-boost list. The upstream NeMo model supports
  n-gram LM fusion and ILM subtraction, so the capability exists in principle.
- Failing that, a **post-STT correction pass** against a small personal lexicon
  (contact names, place names, model names) using fuzzy matching on phonetic distance.
  Cheap, deterministic, and no model changes.
- Keep the lexicon in the repo as a plain list so it is editable without code changes.

Test with `compare_stt.py --hard`, which already contains both failing words and scores
WER with number/punctuation formatting normalized away.
