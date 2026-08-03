# uncensored-local-voice

A fully **local**, fully **uncensored**, real-time **voice agent** for Apple Silicon. Speak to it, it speaks back. No cloud, no API keys, no safety filters. Full-duplex with barge-in — interrupt it mid-reply and it stops.

```
mic ─► AEC ─► VAD ─► STT ─► LLM ─► TTS ─► speakers
```

Everything runs on your machine. The LLM is uncensored. Conversation is full-duplex with barge-in — interrupt mid-reply and it stops, just like a human.

## What it actually does

- **You speak.** Silero VAD detects start/end of your utterance.
- **It hears you cleanly.** WebRTC AEC (via LiveKit) cancels its own voice from the mic so it doesn't feed back, even with built-in mic+speaker.
- **Parakeet TDT v3 STT** transcribes (~69 ms warm).
- **SuperGemma4-26B-Uncensored** (April 2026 release, MoE with ~4 B active params) generates a reply, streamed token-by-token via Ollama.
- Tokens are split into sentences; each sentence is fed to **Kokoro TTS** (MLX-native) immediately. First audio plays while the LLM is still generating sentence 2.
- **Barge-in:** start talking and the agent shuts up. Energy gate + sustain-frame check filters out residual echo.
- **Conversation memory** persists across runs in `.voice_history.json` (rolling 8-turn window).

Measured on **this** machine — M5 MacBook Air (32 GB), 2026-08-03, via
[`scripts/bench_stack.py`](scripts/bench_stack.py), 3 runs per slot. Every published latency
figure I could find for these models came from an M4 Max, an H100, or a vendor blog, so the
repo ships its own benchmark rather than quoting someone else's hardware.

| Stage | Median | Range |
|---|---|---|
| STT — Parakeet TDT v3 | **69 ms** | 67–81 ms |
| LLM — time to first token | **704 ms** | 564–2,088 ms |
| LLM — **time to first sentence** | **2,995 ms** | ← this is what actually gates audio |
| TTS — time to first audio (Kokoro) | **463 ms** | 452–492 ms, 6.4× realtime |
| **Time to first audio, end to end** | **~3.5 s** | this run |

Across six benchmark runs it lands between **~1.3 s and ~3.6 s** depending on thermal state and
what else the machine is doing. An earlier revision of this README claimed "sub-second" — that
was never measured and never met. See [`issues/0003`](issues/0003-llm-decode-rate-dominates-latency.md).

**The non-obvious result: time-to-first-token is not the number that matters.** The first token
arrives in ~300–700 ms, but nothing can be *spoken* until a complete sentence exists, and at
11–18 tok/s that takes 1–3 seconds. Optimising TTFT buys you nothing here — decode rate is the wall.

Which makes the runtime the highest-leverage variable. Same model family, same machine:

| Runtime | Decode | Time to first sentence |
|---|---|---|
| Ollama (GGUF Q4_K_M) | 11–17 tok/s | 977–2,995 ms (high variance) |
| MLX (`mlx-lm`, 4-bit) | **18.3 tok/s** | **~1,200 ms** (consistent) |

## Hardware

Tested on **Apple Silicon (M5, 32 GB)**. Should work on M1–M5 with 16+ GB. Linux/Windows untested — Kokoro MLX is Apple Silicon only; you'd need to swap TTS.

## Stack

| Component | Pick | Why |
|---|---|---|
| LLM | [SuperGemma4-26B-Uncensored](https://huggingface.co/Jiunsong/supergemma4-26b-uncensored-gguf-v2) via [Ollama](https://ollama.com) | Apr 2026, MoE, uncensored fine-tune of Gemma 4 |
| STT | [Parakeet TDT v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) via mlx-audio | Chosen on measured real speech: 13.3% WER / 69 ms, vs Moonshine's 17.8% / 351 ms on the same clips |
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
4. **History is a latency multiplier.** Every turn you keep in context lengthens prefill, and time-to-first-audio degrades badly as the window grows — at ~25 turns it becomes unusable for conversation. The numbers in the table above are measured at a short history; `HISTORY_MAX_TURNS=8` is the cap that keeps it in range. Raise it only if you need long context and can accept the cost.

## Files

```
voice_agent.py             # main pipeline
engines.py                 # pluggable STT/TTS/LLM engine registry
pyproject.toml + uv.lock   # uv-managed deps, Python 3.12
Modelfile.voice            # earlier Ollama alias (kept for reference)

scripts/bench_stack.py     # per-slot latency benchmark — the numbers above
scripts/compare_stt.py     # STT engine comparison on real speech (WER + latency)
scripts/bench_llm_mlx.py   # MLX-runtime LLM benchmark
scripts/smoke_pipeline.py  # E2E test without a mic (uses macOS `say`)
scripts/smoke_turn.py      # single-turn check
scripts/smoke_multiturn.py # multi-turn + history behaviour
scripts/smoke_vad.py       # VAD detection accuracy test

CLAUDE.md                  # architecture + the gotchas, written for whoever works on this next
state.md                   # live state: what's done, what's next, what's blocked
state-log.md               # older session handoffs, newest first
learning.md                # durable lessons — each one caused by a real bug
issues/                    # tracked issues with frontmatter (id/status/priority/area)
```

### Why the context files are in the repo

`CLAUDE.md`, `state.md`, `learning.md` and `issues/` are checked in deliberately, not by
accident. This project is built with coding agents doing most of the typing, and those files
are what make that work: the agent reads `state.md` to know where things stand, appends to
`learning.md` when something breaks, and files an issue instead of derailing.

They are also the honest record. [`issues/0003`](issues/0003-llm-decode-rate-dominates-latency.md)
is where the sub-second claim died, and `learning.md` says why. If you want to know what this
codebase actually does and where it falls short, those files will tell you faster than the source.

## Why uncensored

The LLM is a fine-tuned variant of Gemma 4 with refusal patterns removed. This is **not** because uncensored = good for everything — it's because hardcoded "I cannot…" disclaimers and safety preambles ruin natural voice conversation. A friend on a phone call doesn't say "as an AI I cannot." They just talk. That's the design goal.

Use responsibly. You own anything it says — the model has no built-in safety, so don't deploy it public-facing.

## License

MIT — see [`LICENSE`](LICENSE). Built on top of open-source components (Parakeet TDT, Silero VAD, Kokoro, LiveKit RTC, Ollama, llama.cpp, SuperGemma4) — see their respective licenses.

## Status

Working, actively developed. Last benchmarked 2026-08-03 on an M5 MacBook Air (32 GB).
See [`state.md`](state.md) for the current roadmap and [`issues/`](issues/) for what's known-broken.

Contributions welcome — particularly:

- **Cutting time-to-first-sentence**, which is the real latency wall (see the table above).
  Streaming the first clause to TTS before the sentence completes is the obvious attack.
- Making MLX the default LLM runtime — it already benchmarks faster and more consistently
  than Ollama here, but the engine registry needs the work.
- macOS Voice-Processing-IO (Apple's native AEC) integration via PyObjC
- Adaptive RMS gate (auto-calibrate barge-in threshold)
- Long-history summarization
- Push-to-talk fallback
- Linux/Windows support — Kokoro MLX is Apple-Silicon-only, so the TTS slot needs an alternative
