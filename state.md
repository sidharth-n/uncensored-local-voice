# uncensored-local-voice — State

_Last updated: 2026-07-27_

## Now
- v1 shipped and stable (see `state-log.md` for build history). No code changes this session.
- This session was **research**: found the July 2026 model landscape and upgrade path.
- Branch `main`, clean, pushed.

## Next
1. **Model swap A/B test** — pull `tinyrick/Qwen3.6-35B-A3B-uncensored-heretic-vision-llmfan46:Q4_K_M`
   (~20 GB, MoE 3B-active, ~60–70 tok/s expected vs current ~40) and compare against
   SuperGemma4 with `scripts/smoke_pipeline.py` (TTFT, tok/s, refusals, coherence).
   Fallbacks if the llmfan46 heretic shows coherence drops (one Reddit report):
   `fredrezones55/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive` (69.8K pulls, but 4×
   KL drift per the "Abliterlitics" forensic benchmark) or `HammerAI/gemma-4-26b-a4b-heretic`
   (same base as current, ~16 GB, low-risk). Keep `"think": false` — Qwen3.6 is a
   thinking model too.
2. **TTS slot upgrade** — Qwen3-TTS (Jan 2026, 0.6B/1.7B): 97 ms streaming TTFA,
   3-second voice cloning, voice design. Direct Kokoro replacement candidate; check for
   an MLX port first.
3. **Tool calling for the voice agent** (was a stretch goal; Qwen3.6 has reliable tool
   calling) — voice controls Mac/calendar/files. Then optionally vision ("look at this").
4. Existing tuning roadmap still valid: adaptive RMS gate · truncate barge-in'd partial
   replies in history · macOS VoiceProcessingIO for stronger AEC · long-history
   summarization · push-to-talk · session log · web status UI.

## Blockers
- None. (32 GB note: 20–22 GB Qwen model + Kokoro + Moonshine is tighter than today's
  16 GB — if swap pressure appears, use Q4_K_S or the Gemma-4-26B option.)

## Latest handoff — 2026-07-27 (research session, no code)
- **Uncensored model landscape (researched via agent-reach):** Heretic
  (github.com/p-e-w/heretic, v1.4.0) is now the standard abliteration tool; community
  forensic benchmark ("Abliterlitics", r/LocalLLaMA) showed Heretic builds stay closest
  to base (KL ≈ 0.06) while HauhauCS "aggressive" drifts ~4×. Mac sweet spot for
  32–48 GB is **Qwen3.6-35B-A3B** (MoE, 3B active, 256K ctx, vision, tool calling);
  uncensored GGUFs pullable from Ollama (tags in Next #1). OMLX + MTP reaches ~70 tok/s.
- **Local capability map for this Mac (32 GB M5):** agents (Hermes Agent on local
  Ollama — 24/7 Telegram assistant), local coding agents (Claude Code/OpenCode via
  claude-code-router; context is the wall, ~64K working budget), overnight batch
  (~1.4K summarizations/8 h for ~$0.14 power), private RAG, image gen (Draw Things /
  Flux), music gen (ACE-Step 1.5), speech-to-speech (Moshi MLX). Video gen is the weak
  slot (Wan 2.2 5B ≈ 47–97 min per 5 s clip on 32 GB). Modality details captured in the
  2026-07-27 conversation; roadmap idea: formalize voice_agent stages as swappable
  slots + tool belt.
- **Spun off a new project: `livefunAI`** (~/Developer/Personal/livefunAI, private
  GitHub) — live AI event entertainment (Decart realtime restyle + fal.ai I2V clips on
  a video wall). Scaffolded, registered in the brain, MVP plan approved, Stage 0
  (operator+wall shell, camera passthrough) built/verified/pushed. Work continues in
  its own iTerm tab/session — not in this repo.
- **First next session here:** run the model A/B (Next #1); it's a pure
  `ollama pull` + env/model-name change + smoke test.
