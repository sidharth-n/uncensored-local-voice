# uncensored-local-voice

A fully local, fully uncensored, real-time voice agent. Mic in → spoken reply out. No cloud, no safety filters, latency targeted at <1 s warm.

## Pipeline

```
mic 16kHz/10ms ─► WebRTC AEC (livekit.rtc) ─► Silero VAD ─► Moonshine STT
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

## Stack

| Component | Pick | Why |
|---|---|---|
| LLM | `0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M` via **Ollama** | Apr 2026, MoE 4B-active, uncensored, ~16 GB RAM |
| STT | **Moonshine** `MEDIUM_STREAMING` (ONNX) | Streaming-built, ~75 ms warm |
| VAD | **Silero VAD** `VADIterator` | Industry standard, 32 ms frames |
| AEC | **`livekit.rtc.AudioProcessingModule`** (WebRTC AEC3) | Production-grade, pip-installable, 10 ms frames @ 16 kHz |
| TTS | **`kokoro-mlx`** (Apple Silicon native MLX) | Per-sentence streaming, 17× realtime |
| Audio I/O | **sounddevice** | PortAudio bindings |

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
| `VAD_THRESHOLD` | `0.6` | Silero VAD speech probability cutoff. |
| `HISTORY_MAX_TURNS` | `8` | Rolling user/assistant pair window. Keep low to keep TTFA fast. |

## Critical gotchas

1. **`think: false` is non-negotiable.** SuperGemma4 ships thinking-mode ON; with it on, the response field stays empty for 5–30 s while the model dumps reasoning into a `thinking` field. Voice agents cannot tolerate that. The Modelfile cannot disable it (`PARAMETER think false` is unsupported); each `/api/chat` request must include `"think": false`.
2. **WebRTC AEC requires exactly 10 ms frames at 16 kHz** (160 samples, int16). Both mic capture and the TTS reverse stream must conform — any other frame size silently fails.
3. **Kokoro outputs at 24 kHz** — we resample to 16 kHz for both speaker output and AEC reverse stream so they match. Different rates would break AEC.
4. **AEC has trouble with built-in MacBook mic+speaker geometry.** WebRTC's AEC3 isn't perfect during double-talk; some residual leaks. The energy gate (`BARGE_IN_RMS_GATE`) + sustain-frame check is what makes barge-in reliable. Headphones eliminate the problem entirely.
5. **History balloons → TTFA balloons.** 20 turns can push TTFA past 5 s. The cap (`HISTORY_MAX_TURNS=8`) is enforced both at load and after every turn.

## Files

```
voice_agent.py           # main pipeline (~430 LOC)
scripts/smoke_pipeline.py # E2E test without mic (uses macOS `say`)
scripts/smoke_vad.py      # VAD detection accuracy test
pyproject.toml            # uv-managed, Python 3.12 (kokoro-mlx requires 3.10–3.12)
.voice_history.json       # rolling conversation memory (auto-managed)
Modelfile.voice           # earlier Ollama alias (kept for reference, unused by agent)
```

## Quick handoff

For a clean restart: `rm .voice_history.json`. The model stays warm in Ollama for 30 min between runs (`keep_alive: 30m`).

For session continuation: see `state.md` — `/start` reads it, `/end` updates it.
