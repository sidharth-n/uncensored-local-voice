---
id: 0003
title: LLM decode rate dominates latency — swap to a 3B-active MoE
status: open
priority: urgent
area: llm
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

The LLM is the largest remaining latency cost. Time-to-first-audio sat at
2248-3647 ms in live use, and roughly a second of that is the model producing its
first sentence.

## Why it matters

Every other slot is now small: STT 64-230 ms, TTS TTFA ~300 ms, turn detection 19 ms.
Nothing else on the roadmap buys as much.

## Evidence

`scripts/bench_stack.py`, current model
(`0xIbra/supergemma4-26b-uncensored-gguf-v2:Q4_K_M`):

```
TTFT            318.6 ms
first sentence  976.9 ms   <- gates audio
decode           16.1 tok/s
```

TTFT is **not** the metric — TTS is driven per sentence, so nothing is audible until
the first sentence terminator arrives, which is a function of decode rate. At
16.1 tok/s a ~25-token sentence costs ~1.5 s.

Research 2026-07-30: `Qwen3.6-35B-A3B` (MoE, 3B active) measured at **61.2 tok/s** on
M1 Max vs 16.7 tok/s for a dense 27B. ~3.7x would put first-sentence near 400 ms.

Candidates verified live on ollama.com (HTTP 200, fake tag control returns 404):

| Tag | Size | Note |
|---|---|---|
| `huihui_ai/Qwen3.6-abliterated` | 17 GB | best on *both* KL drift (0.0074) and refusal removal (98.5% ASR) per the independent Abliterlitics run; same footprint as today |
| `tinyrick/Qwen3.6-35B-A3B-uncensored-heretic-vision-llmfan46:Q4_K_M` | 22 GB | lowest KL (0.0037), adds vision; tighter on 32 GB |
| `HammerAI/gemma-4-26b-a4b-heretic` | 17 GB | low-risk, same base family as current |

Rejected: `fredrezones55/...HauhauCS-Aggressive` — 6.5x Heretic's KL drift and built on
a tool plagiarised from Heretic (AGPL violation confirmed by Heretic's author).

## Approach

1. `ollama pull huihui_ai/Qwen3.6-abliterated` and A/B with
   `bench_stack.py --slot llm --llm <a> --llm <b>`, holding `think: false` constant.
2. Measure first-sentence latency and decode rate, not TTFT.
3. **Do not enable MTP / speculative decoding** — a net loss on Metal: baseline
   Qwen3.5-9B 25.3 tok/s drops to 19.3, and Qwen3.6-35B self-MTP collapses to
   1.93 tok/s (llama.cpp issues #23752, #23011).
4. Worth testing MLX vs Ollama for the same model: MLX is 20-30% faster on decode,
   which is the gate here, though Ollama/llama.cpp holds the TTFT crown. Note this
   machine runs Ollama 0.21.2; newer releases added an MLX backend (~2x decode) and
   fixed an MLX memory leak, so a runtime upgrade may beat a model swap.
5. Reversible: one env var / `LLM_MODEL` change.
