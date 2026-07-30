# uncensored-local-voice — State

_Last updated: 2026-07-30_

## Now
- Branch **`voice-stack-upgrade-2026-07`** (9 commits, all pushed). `main` untouched.
- **Known-good config is `TURN_DETECTOR=off`** — Sid confirmed it works. Run it that way.
- Stack now: **Parakeet TDT v3** STT (was Moonshine) · Kokoro TTS · Silero VAD
  (min_silence 500 ms with detector off) · WebRTC AEC · SuperGemma4 LLM (untouched).
- New: `issues/` tracker with **8 open issues** (3 urgent) — read `issues/README.md` first.
- Last verified: 6/6 `scripts/smoke_multiturn.py`, full `smoke_pipeline.py` E2E, and a
  real 10-turn conversation by Sid.

## Next
1. **`issues/0003` — LLM swap.** Biggest remaining win and lowest risk: one-line,
   reversible. `ollama pull huihui_ai/Qwen3.6-abliterated` (17 GB, same footprint as
   today), then `uv run python scripts/bench_stack.py --slot llm --llm <old> --llm <new>`.
   Target: first-sentence 977 ms → ~400 ms. **Do not enable MTP** (net loss on Metal).
2. **`issues/0002` — inter-sentence gap.** Callback-driven output stream + ring buffer so
   TTS generates ahead of playback. Must move the AEC reverse-stream feed carefully.
3. **`issues/0001` — barge-in.** Do after 0002; they share the playback mechanism.
4. `issues/0004` (STT slower in-pipeline than isolated — may be free latency), then
   0007 / 0005 / 0006 / 0008.

## Blockers
- None. Nothing is waiting on a decision.

## Latest handoff — 2026-07-30 (research + 9 commits, agent materially better but not "natural" yet)

### What shipped
Full research pass (agent-reach ×5 + monid/tikhub X pulls, $0.0165 spend) then a staged
rebuild. Every number below was measured on **this M5**, not taken from a vendor page.

| Slot | Before | After |
|---|---|---|
| STT | Moonshine, 396 ms, 17.8% WER | **Parakeet TDT v3, 74 ms isolated, 13.3% WER** |
| Turn-taking | silence timeout only | Smart Turn v3.2 available (19 ms) — **default off** |
| Threading | thread per turn | **one persistent worker** |
| Barge-in | fixed RMS gate, no VAD check | VAD-gated + output-scaled gate (still wrong, see 0001) |

Commits: `08c87ea` mlx-audio added · `abbfc17` engine slots · `e0874ce` bench harness ·
`a04e245` Smart Turn · `de96969` fragmentation fix + STT engines · `71efaef` Parakeet
default · `b8be6e1` GIL crash fix + multi-turn test · `20dfa9e` self-interruption /
history / utterance cap · `0df0154` continuation bound + audio-length logging.

### Key decisions and the evidence
- **`mlx-audio` (⭐7650) is the single runtime for STT+TTS+VAD** — replaced what would
  have been four separate packages. Installs clean on Python 3.13, so no venv rebuild.
- **Parakeet over Nemotron/Whisper/Moonshine**, decided on a hard 45-word passage read
  by Sid: nemotron 8.9% WER / 2174 ms, **parakeet 13.3% / 228 ms**, whisper 15.6% /
  1183 ms, moonshine 17.8% / 1682 ms. Nemotron's margin is partly a scoring artifact
  (it split the place name into two words = 2 errors for one mistake) and its 2.1 s
  exceeds the whole latency budget. Parakeet was also the only engine to get "just buy
  it" right. `STT_ENGINE=nemotron` remains available.
- **The metric that matters is time-to-first-sentence, not TTFT** — TTS is driven per
  sentence, so nothing is audible until a terminator arrives. TTFT 319 ms vs
  first-sentence 977 ms; the gap is decode rate (16.1 tok/s). This reordered the plan.
- **Malayalam deliberately parked** by Sid — findings preserved in `issues/0006`.

### Corrections to earlier project claims (all were wrong in `CLAUDE.md`/`state.md`)
- STT "~75 ms warm" → measured **351–396 ms** (Moonshine).
- Kokoro "17× realtime" → measured **8.9–14.7×**, TTFA ~300 ms.
- "MTP reaches ~70 tok/s" → **MTP is a net loss on Metal** (35B self-MTP collapses to
  1.93 tok/s).
- HauhauCS "~4× KL drift" → **6.5×**, and its tool is plagiarised from Heretic.
- `<1 s warm` target → actual ~1.6 s isolated, 2.2–3.6 s TTFA live.
- **`CLAUDE.md` still carries the wrong STT/TTS/MTP numbers — fix it next session.**

### Process lesson (this cost Sid five test sessions)
Three of my changes shipped clean through scripted tests and broke in live conversation:
utterance fragmentation, a fatal GIL crash, and runaway buffers. Cause: component tests
(`smoke_pipeline.py` = one turn, main thread) cannot catch interaction bugs.
`scripts/smoke_multiturn.py` now covers multi-turn threading, but the real fix is
**record conversations and replay offline** before enabling anything conversational —
see `issues/0005`. Do not develop turn-taking by shipping to Sid and reading logs.

### First thing next session
Read `issues/README.md`. Then do `0003` (LLM swap) — it is isolated from the audio path,
one env var to revert, and worth ~1 s off every reply. Leave `TURN_DETECTOR` off.
