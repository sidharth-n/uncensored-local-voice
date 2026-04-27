# uncensored-local-voice

A fully **local**, fully **uncensored**, real-time **voice agent** for Apple Silicon. Speak to it, it speaks back. No cloud, no API keys, no safety filters. Sub-second time-to-first-audio on a MacBook Air.

```
mic ─► AEC ─► VAD ─► STT ─► LLM ─► TTS ─► speakers
```

Everything runs on your machine. The LLM is uncensored. Conversation is full-duplex with barge-in — interrupt mid-reply and it stops, just like a human.

## What it actually does

- **You speak.** Silero VAD detects start/end of your utterance.
- **It hears you cleanly.** WebRTC AEC (via LiveKit) cancels its own voice from the mic so it doesn't feed back, even with built-in mic+speaker.
- **Moonshine STT** transcribes (~75 ms warm).
- **SuperGemma4-26B-Uncensored** (April 2026 release, MoE with ~4 B active params) generates a reply, streamed token-by-token via Ollama.
- Tokens are split into sentences; each sentence is fed to **Kokoro TTS** (MLX-native) immediately. First audio plays while the LLM is still generating sentence 2.
- **Barge-in:** start talking and the agent shuts up. Energy gate + sustain-frame check filters out residual echo.
- **Conversation memory** persists across runs in `.voice_history.json` (rolling 8-turn window).

Measured warm baseline on M5 MacBook Air (32 GB):

| Stage | Time |
|---|---|
| STT | ~75 ms |
| LLM TTFT | ~720 ms |
| TTS TTFA | ~360 ms |
| **End-to-end** | **~1.16 s** |

## Hardware

Tested on **Apple Silicon (M5, 32 GB)**. Should work on M1–M5 with 16+ GB. Linux/Windows untested — Kokoro MLX is Apple Silicon only; you'd need to swap TTS.

## Stack

| Component | Pick | Why |
|---|---|---|
| LLM | [SuperGemma4-26B-Uncensored](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2) via [Ollama](https://ollama.com) | Apr 2026, MoE, uncensored fine-tune of Gemma 4 |
| STT | [Moonshine](https://github.com/moonshine-ai/moonshine) `MEDIUM_STREAMING` | Built for streaming, ~75 ms on Apple Silicon |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) | 1 MB ONNX, industry standard |
| AEC | [`livekit.rtc.AudioProcessingModule`](https://docs.livekit.io/reference/python/v1/livekit/rtc/apm.html) (WebRTC AEC3) | Production-grade, pip-installable, standalone |
| TTS | [`kokoro-mlx`](https://pypi.org/project/kokoro-mlx/) | MLX-native, per-sentence streaming, 17× realtime |
| Audio I/O | [sounddevice](https://python-sounddevice.readthedocs.io/) | PortAudio bindings, low-latency callbacks |
| Runtime | [uv](https://github.com/astral-sh/uv) + Python 3.12 | kokoro-mlx requires 3.10–3.12 |

## Setup

```bash
# 1. Install Ollama
brew install ollama
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve &

# 2. Pull the LLM (~16 GB)
ollama pull 0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M

# 3. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 4. Clone + install Python deps
git clone https://github.com/sidharth-n/uncensored-local-voice.git
cd uncensored-local-voice
uv python install 3.12
uv sync   # installs everything from pyproject.toml + uv.lock

# 5. Run
uv run python voice_agent.py
```

You'll see `=== ready (FULL-DUPLEX + AEC + barge-in) ===` after ~5–10 s on first cold start. Then just talk. `Ctrl+C` to quit. Conversation persists to `.voice_history.json`.

## Modes

```bash
# default — full-duplex with AEC + barge-in
uv run python voice_agent.py

# headphones / quieter room (more aggressive barge-in)
HEADPHONES=1 BARGE_IN_RMS_GATE=0.04 BARGE_IN_SUSTAIN_FRAMES=3 uv run python voice_agent.py

# noisy room or echo-prone speakers (no barge-in but no echo loop either)
HALF_DUPLEX=1 uv run python voice_agent.py
```

## Tunable env vars

| Var | Default | Effect |
|---|---|---|
| `HALF_DUPLEX` | `0` | `1` disables AEC; mic muted while TTS plays |
| `STREAM_DELAY_MS` | `80` | AEC speaker→mic round-trip estimate. Raise (100–150) if echo bleeds; lower (40–60) if first words clip |
| `BARGE_IN_RMS_GATE` | `0.05` | RMS floor on cleaned mic to count as user voice |
| `BARGE_IN_SUSTAIN_FRAMES` | `4` | Consecutive 32 ms frames above gate before barge-in fires |
| `VAD_THRESHOLD` | `0.6` | Silero VAD speech probability |
| `HISTORY_MAX_TURNS` | `8` | Rolling user/assistant pair window |

## Critical gotchas

1. **`think: false` is non-negotiable.** SuperGemma4 ships thinking-mode ON; with it on the response field stays empty for 5–30 s while reasoning streams to a separate `thinking` field. Modelfile cannot disable it ([ollama#14809](https://github.com/ollama/ollama/issues/14809)). Every `/api/chat` call from the agent passes `"think": false`.
2. **AEC frames must be exactly 10 ms at 16 kHz.** WebRTC APM is strict — wrong frame size fails silently.
3. **Built-in MacBook mic+speaker geometry is hard for software AEC.** Expect to tune the env vars for your room. Headphones eliminate the problem.
4. **History is the latency killer.** TTFA goes from 700 ms at 0 turns to 5+ s at 25 turns. Cap (`HISTORY_MAX_TURNS=8`) prevents this; raise it only if you need long context.

## Files

```
voice_agent.py            # main pipeline
scripts/smoke_pipeline.py # E2E test without mic (uses macOS `say`)
scripts/smoke_vad.py      # VAD detection accuracy test
pyproject.toml + uv.lock  # uv-managed deps, Python 3.12
CLAUDE.md                 # internal architecture notes
state.md                  # session handoff (what's done / next)
Modelfile.voice           # earlier Ollama alias (kept for reference)
```

## Why uncensored

The LLM is a fine-tuned variant of Gemma 4 with refusal patterns removed. This is **not** because uncensored = good for everything — it's because hardcoded "I cannot…" disclaimers and safety preambles ruin natural voice conversation. A friend on a phone call doesn't say "as an AI I cannot." They just talk. That's the design goal.

Use responsibly. You own anything it says — the model has no built-in safety, so don't deploy it public-facing.

## License

MIT. Built on top of open-source components (Moonshine, Silero VAD, Kokoro, LiveKit RTC, Ollama, llama.cpp, SuperGemma4) — see their respective licenses.

## Status

Working v1 as of April 2026. See [`state.md`](state.md) for current roadmap. Contributions welcome — particularly:

- macOS Voice-Processing-IO (Apple's native AEC) integration via PyObjC
- Adaptive RMS gate (auto-calibrate barge-in threshold)
- Long-history summarization
- Push-to-talk fallback
