# uncensored-local-voice

A fully local, fully uncensored, real-time voice agent. Mic in → spoken reply out. No cloud, no safety filters.

**Measured latency (2026-07-30, this M5):** ~1.6 s isolated, 2.2–3.6 s time-to-first-audio in
live conversation. The original "<1 s warm" claim was never met — see `issues/0003`, the LLM's
first-sentence time is the dominant cost.

## Pipeline

```
mic 16kHz/10ms ─► WebRTC AEC (livekit.rtc) ─► Silero VAD ─► Parakeet TDT v3 STT
                            ▲                                        │
                            │ reverse-stream reference               ▼
                  ┌─ resampled 24→16kHz ──┐                  Ollama /api/chat
                  │                       │                  (think:false,
                  │                       │                   keep_alive:30m)
              speakers ◄── KokoroTTS ─────┘                          │
                                                          sentence_stream
                                                                    │
                                                                    ▼
                                                              KokoroTTS
                                                              (per-sentence)
```

Every engine slot is swappable by env var — see `engines.py`.

## Stack

| Component | Pick | Why |
|---|---|---|
| LLM | `0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` via **Ollama** | Apr 2026, MoE 26B/4B-active, uncensored, ~19 GB resident. Measured **14–16 tok/s**, first sentence ~1.0–1.3 s — the pipeline's biggest remaining cost |
| STT | **Parakeet TDT v3** (`mlx-community/parakeet-tdt-0.6b-v3`) | Chosen on real speech: 13.3% WER / 74 ms isolated, vs Moonshine 17.8% / 396 ms |
| VAD | **Silero VAD** `VADIterator` | Industry standard, 32 ms frames |
| AEC | **`livekit.rtc.AudioProcessingModule`** (WebRTC AEC3) | Production-grade, pip-installable, 10 ms frames @ 16 kHz |
| TTS | **`kokoro-mlx`** (Apple Silicon native MLX) | Per-sentence streaming. Measured **8.9–14.7× realtime**, TTFA ~300 ms |
| Runtime | **`mlx-audio`** | One MLX runtime for STT + TTS + turn detection instead of four packages |
| Audio I/O | **sounddevice** | PortAudio bindings |

Available alternates: `STT_ENGINE` = `parakeet` (default) · `nemotron` · `nemotron-8bit` ·
`whisper` · `moonshine` · any mlx-audio repo id. `TTS_ENGINE` = `kokoro`.
`TURN_DETECTOR` = `off` (default) · `smartturn`.

## Run

```bash
cd ~/Developer/Personal/uncensored-local-voice

# default: full-duplex with AEC + barge-in
uv run python voice_agent.py

# headphones / quieter room — more aggressive barge-in
HEADPHONES=1 BARGE_IN_RMS_GATE=0.04 BARGE_IN_SUSTAIN_FRAMES=3 uv run python voice_agent.py

# noisy room or echo-y speakers — fall back to half-duplex (no barge-in, no echo loop)
HALF_DUPLEX=1 uv run python voice_agent.py
```

## Tunable env vars

| Var | Default | Effect |
|---|---|---|
| `HALF_DUPLEX` | `0` | `1` disables AEC; mic muted while TTS plays. No barge-in. |
| `STREAM_DELAY_MS` | `80` | Speaker→mic round-trip estimate for AEC. Raise (100–150) if echo bleeds; lower (40–60) if first words get clipped. |
| `BARGE_IN_RMS_GATE` | `0.05` | RMS floor of cleaned mic audio to count as user voice during TTS. |
| `BARGE_IN_SUSTAIN_FRAMES` | `4` | Consecutive 32 ms frames above gate before barge-in fires. |
| `BARGE_IN_ECHO_FACTOR` | `1.6` | Gate is scaled by this × current output level while TTS plays. |
| `VAD_THRESHOLD` | `0.6` | Silero VAD speech probability cutoff. |
| `VAD_MIN_SILENCE_MS` | `500` (detector off) | Silence before an utterance is considered ended. |
| `MIN_UTTERANCE_MS` | `250` | Shorter trips are discarded as noise, not transcribed. |
| `MAX_UTTERANCE_S` | `12` | Hard ceiling on a single buffered utterance. |
| `MAX_TURN_CONTINUATIONS` | `2` | Times semantic turn detection may extend one turn. |
| `HISTORY_MAX_TURNS` | `8` | Rolling user/assistant pair window. Keep low to keep TTFA fast. |
| `STT_ENGINE` | `parakeet` | See stack table for alternates. |
| `TTS_ENGINE` | `kokoro` | — |
| `TURN_DETECTOR` | `off` | `smartturn` enables Smart Turn v3.2 semantic end-of-turn. **Leave off** — see `issues/0005`. |

## Critical gotchas

1. **`think: false` is non-negotiable.** SuperGemma4 ships thinking-mode ON; with it on, the response field stays empty for 5–30 s while the model dumps reasoning into a `thinking` field. Voice agents cannot tolerate that. The Modelfile cannot disable it (`PARAMETER think false` is unsupported); each `/api/chat` request must include `"think": false`.
2. **WebRTC AEC requires exactly 10 ms frames at 16 kHz** (160 samples, int16). Both mic capture and the TTS reverse stream must conform — any other frame size silently fails.
3. **Kokoro outputs at 24 kHz** — we resample to 16 kHz for both speaker output and AEC reverse stream so they match. Different rates would break AEC.
4. **AEC has trouble with built-in MacBook mic+speaker geometry.** WebRTC's AEC3 isn't perfect during double-talk; some residual leaks. The energy gate (`BARGE_IN_RMS_GATE`) + sustain-frame check is what makes barge-in reliable. Headphones eliminate the problem entirely.
5. **History balloons → TTFA balloons.** 20 turns can push TTFA past 5 s. The cap (`HISTORY_MAX_TURNS=8`) is enforced both at load and after every turn.
6. **One persistent worker thread, never one per turn.** MLX keeps thread-local Metal state; destroying it on thread exit kills the interpreter (`PyThreadState_Get: ... the GIL is released`). This was fatal on turn 2 once a second MLX model was added.
7. **Published latency numbers are useless here — measure on this machine.** `scripts/bench_stack.py` (per slot) and `scripts/compare_stt.py` (real speech, WER-scored) exist for that. The metric that matters is **time-to-first-sentence**, not TTFT: TTS is driven per sentence, and the two differ ~3–4× on the same model.
8. **Component tests cannot catch conversation bugs.** Anything touching turn-taking, buffering or threading must be exercised by `scripts/smoke_multiturn.py` (multi-turn + worker thread). `smoke_pipeline.py` runs one turn on the main thread and is structurally blind to this class of bug.

## Files

```
voice_agent.py            # main pipeline (~730 LOC)
engines.py                # swappable STT / TTS / turn-detector slots (~380 LOC)
scripts/bench_stack.py    # per-slot latency benchmark (STT / LLM / TTS)
scripts/bench_llm_mlx.py  # same LLM metrics for an MLX-format model
scripts/compare_stt.py    # WER-scored STT comparison on real speech
scripts/smoke_multiturn.py # multi-turn + worker-thread regression test
scripts/smoke_pipeline.py # E2E test without mic (uses macOS `say`)
scripts/smoke_turn.py     # semantic turn-detector test
scripts/smoke_vad.py      # VAD detection accuracy test
bench_results.jsonl       # accumulated benchmark rows, comparable across runs
issues/                   # local issue tracker — read issues/README.md
pyproject.toml            # uv-managed, Python >=3.12
.voice_history.json       # rolling conversation memory (auto-managed)
Modelfile.voice           # earlier Ollama alias (kept for reference, unused by agent)
```

## Quick handoff

For a clean restart: `rm .voice_history.json`. The model stays warm in Ollama for 30 min between runs (`keep_alive: 30m`).

For session continuation: see `state.md` — `/start` reads it, `/end` updates it.
