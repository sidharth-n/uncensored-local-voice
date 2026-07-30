---
id: 0006
title: Malayalam support parked until the English path is solid
status: open
priority: low
area: multilingual
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Malayalam (and Indic languages generally) were researched in depth, then deliberately
deferred. Sid: *"let's not focus more on Malayalam, we can add that later... first make
our existing model for English the best out there."*

## Why it matters

Nothing is blocked by this, but the research cost real effort and the findings should
not have to be rediscovered.

## Evidence — findings worth keeping

**TTS.** `k2-fsa/OmniVoice` (in `mlx-audio`, 860k downloads) lists **644 language
codes** including `ml`, plus Tamil, Telugu, Kannada, Hindi, Bengali, Marathi, Gujarati,
Punjabi, Urdu, Assamese, Sanskrit, Sinhala and Tulu. But its Malayalam text
normalization is **unmerged**: PR
[k2-fsa/OmniVoice#161](https://github.com/k2-fsa/OmniVoice/pull/161) (open since
2026-05-19) states *"Without this, the model tries to pronounce raw symbols like ₹, %,
5:30 which sounds broken."* Usable for prose, broken for times/currency/units until
that PR is applied manually.

Better on quality evidence, both MIT: **AI4Bharat IndicF5** (0.4B, 11 languages,
1417 hrs, needs reference audio + transcript) and **ai4bharat/indic-parler-tts**
(21 languages, 1806 hrs, emotion-conditioned).

Qwen3-TTS covers only 10 languages, **no Indic at all** — it is not an option if
Malayalam matters.

**STT.** Nemotron 3.5 ASR's 40 locales include only `hi-IN` for Indic — **no Malayalam**.
Parakeet TDT v3 is 25 European languages — also no Malayalam. Published Malayalam WER:

| Model | Malayalam WER | License |
|---|---|---|
| IndicWhisper (AI4Bharat) | **32.3% avg** (beats Google 47.9%, Azure 41.8%) | — |
| MMS-1b-all (in mlx-audio) | 21.2% (Kathbath) | CC-BY-**NC** |
| thennal/whisper-medium-ml | 11.49% normalized / 38.62% raw | Apache-2.0 |
| IndicConformer-600M | no public number | MIT |

Stock Whisper on Malayalam is ~108% WER — unusable. On a harder benchmark
(spontaneous/noisy) IndicWhisper drops to 48.6%, so expect real-world worse than the
clean numbers.

**LLM.** The one human-eval benchmark found (BhashaKritika, arXiv 2511.10338, Table 16)
scores Malayalam generation **Qwen-3 32B 3.30/3.14 vs Gemma-3 27B 2.86/2.04** — i.e.
Qwen beats Gemma, contradicting the usual "Gemma is the Indic one" prior. Caveat: the
paper's authors excluded Qwen-3 from production over *"high percentage of sentence
repetitions"* in Indic generation. No Malayalam benchmark exists for Qwen3.6, Gemma-4,
or **any abliterated checkpoint** — whether abliteration damages Indic ability is
genuinely unmeasured.

**Turn detection.** Malayalam is not among Smart Turn's 23 languages, and the model
reads lexical content rather than pure prosody, so generalization is unlikely. See
[[0005]].

## Approach

Revive only after [[0001]], [[0002]] and [[0003]] are done. Then:

1. Record ~10 s of real Malayalam speech (macOS `say` has no Malayalam voice, so no
   synthetic probe is possible) and score IndicConformer-600M vs MMS-1b-all with
   `compare_stt.py --reference`.
2. Expect STT to need a **per-language engine**, not one model for both — the slots
   already support this via `STT_ENGINE`.
3. Test `huihui_ai/Qwen3.6-abliterated` vs `HammerAI/gemma-4-26b-a4b-heretic` on ~10
   Malayalam prompts by eye; no benchmark can be borrowed.
