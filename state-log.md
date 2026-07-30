# State Log

Older session handoffs, newest first. Read on demand — `/start` only reads `state.md`.

## 2026-07-27 — Research session (no code)
- **Uncensored model landscape (researched via agent-reach):** Heretic
  (github.com/p-e-w/heretic, v1.4.0) is now the standard abliteration tool; community
  forensic benchmark ("Abliterlitics", r/LocalLLaMA) showed Heretic builds stay closest
  to base (KL ≈ 0.06) while HauhauCS "aggressive" drifts ~4×. Mac sweet spot for
  32–48 GB is **Qwen3.6-35B-A3B** (MoE, 3B active, 256K ctx, vision, tool calling);
  uncensored GGUFs pullable from Ollama. OMLX + MTP reaches ~70 tok/s.
  — *Corrected 2026-07-30: HauhauCS drift is 6.5× not 4×, and MTP is a net LOSS on
  Metal. See `learning.md`.*
- **Local capability map for this Mac (32 GB M5):** agents (Hermes Agent on local
  Ollama), local coding agents (context is the wall, ~64K working budget), overnight
  batch (~1.4K summarizations/8 h for ~$0.14 power), private RAG, image gen (Draw
  Things / Flux), music gen (ACE-Step 1.5), speech-to-speech (Moshi MLX). Video gen is
  the weak slot (Wan 2.2 5B ≈ 47–97 min per 5 s clip on 32 GB).
- **Spun off a new project: `livefunAI`** (~/Developer/Personal/livefunAI, private
  GitHub) — live AI event entertainment (Decart realtime restyle + fal.ai I2V clips on
  a video wall). Scaffolded, registered in the brain, MVP plan approved, Stage 0
  (operator+wall shell, camera passthrough) built/verified/pushed. Work continues in
  its own iTerm tab/session — not in this repo.
- Planned next: model A/B swap (pure `ollama pull` + env change + smoke test), TTS slot
  upgrade to Qwen3-TTS, tool calling for the voice agent.

## 2026-04-27 — Open-sourced on GitHub
- Wrote public-facing `README.md` (setup, modes, env-var reference, hardware notes, gotchas, contribution asks).
- Wrote `.gitignore` excluding `.venv/`, `__pycache__/`, `.voice_history.json` (private convo memory), audio test artifacts, logs.
- `git init -b main`, staged 10 files, initial commit with co-author trailer.
- Created public GitHub repo via `gh repo create`: **https://github.com/sidharth-n/uncensored-local-voice**
- Pushed `main`. Repo description set, branch tracks origin.
- Convention going forward: commit + push regularly during future sessions; `/end` should also push state.md updates.

## 2026-04-27 — Voice agent v1: AEC, barge-in, conversational replies
- Researched + selected **SuperGemma4-26B-Uncensored** (Apr 2026 MoE, ~4B active) as the LLM. Pulled `0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` via Ollama. Smoke-tested: ~40 tok/s, no refusals.
- Discovered Gemma 4 ships with **thinking mode ON** — first chat returned 80 tokens of empty `content`, all CoT in `thinking` field. Hard requirement to send `"think": false` in every `/api/chat` request (Modelfile param not supported yet, see [ollama/ollama#14809](https://github.com/ollama/ollama/issues/14809)).
- Built `voice_agent.py` (~430 LOC): mic → Silero VAD → Moonshine STT → Ollama (think=false, stream=true, keep_alive=30m) → sentence-level split → KokoroTTS (MLX) → speakers. Sentence-level streaming gives sub-second TTFA.
- Added `scripts/smoke_pipeline.py` (E2E without mic via macOS `say`) and `scripts/smoke_vad.py` (Silero accuracy check). Warm baseline: STT 75 ms / LLM TTFT 724 ms / TTS TTFA 363 ms / total 1.16 s.
- Hit acoustic feedback loop on first live test (agent's TTS → speakers → mic → STT → loop). Implemented half-duplex (mic dropped during TTS + 250 ms grace) — fixed echo but killed barge-in.
- Researched 2026 best practice: **WebRTC AEC3** via `livekit.rtc.AudioProcessingModule` is the standard for full-duplex local agents (Pipecat, LiveKit, RealtimeSTT all use it). Wired it in: 10 ms frames at 16 kHz int16, Kokoro 24 kHz output resampled to 16 kHz so reverse stream matches speaker output, `set_stream_delay_ms(80)`.
- Tuned the AEC pipeline iteratively against MacBook Air built-in mic+speaker geometry:
  - First with NS+AGC: AEC residual fully suppressed but user voice was eaten too — Moonshine returned empty for real speech.
  - AEC-only: user voice intact but residual triggered VAD as false barge-ins.
  - Final: **AEC + HPF, no NS, no AGC**, plus an **energy gate** (`BARGE_IN_RMS_GATE=0.05`) and **sustain-frame check** (`BARGE_IN_SUSTAIN_FRAMES=4`, ~128 ms of continuous voice required) before firing barge-in. `barge_fired` flag disarms further fires until the next reply starts.
- Upgraded Moonshine from `BASE` → `MEDIUM_STREAMING` for accuracy. Disabling NS in APM also visibly cleaned up STT garbling.
- Capped history (`HISTORY_MAX_TURNS=8`, env-tunable) and enforced cap at load too — TTFA was hitting 5–8 s once history grew past 20 turns. Now stays under 1.5 s warm regardless of session length.
- Persistent context memory: `.voice_history.json` saves user/assistant turns; rolling-window trim. Each session resumes prior turns automatically.
- Live conversation tested for ~30 turns: real barge-ins fire correctly (RMS 0.07–0.15, sustained), Malayalam reply played through cleanly, conversational personality good ("Hmm", "Haha", contractions, no disclaimers).
- Wrote `CLAUDE.md` with full architecture, run instructions, env-var reference, and gotchas.
