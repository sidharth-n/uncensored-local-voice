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

## Update 2026-07-30 — the premise above is wrong; it's the runtime, not the model

**The current model is already a 4B-active MoE.** From its own GGUF metadata:
`gemma4.expert_count = 128`, `gemma4.expert_used_count = 8`, 30 blocks, and
`ollama ps` reports `100% GPU` at 19 GB. A sparse MoE decoding at 12–16 tok/s is
dense-27B speed. So "swap to a sparse MoE" cannot be the fix — we already have one.

**The likely cause is that our model is a GGUF, so Ollama routes it to
llama.cpp.** This machine's Homebrew Ollama ships an MLX runner
(`/opt/homebrew/Cellar/ollama/0.21.2_1/libexec/lib/ollama/mlx_metal_v3/libmlxc.dylib`)
that only MLX-format models can reach. Also relevant: we run **0.21.2, latest is
0.32.5**, and v0.31.1 shipped "Tightened Gemma 4 MoE model loading in the MLX
engine" plus a new small-batch matmul kernel — both aimed squarely at our case.

**Control experiment (run, conclusive).** Same model class, same runtime, a
different GGUF changes nothing:

| Model (both GGUF / llama.cpp) | TTFT | first sentence | decode |
|---|---|---|---|
| `0xIbra/supergemma4-26b...:Q4_K_M` (current) | 534 ms | 1864 ms | 12.5 tok/s |
| `HammerAI/gemma-4-26b-a4b-heretic` | 441 ms | 2013 ms | 13.0 tok/s |

Within noise. Both depressed by a concurrent download; a clean re-run is in
`bench_results.jsonl`. **Swapping GGUF→GGUF is a dead end — do not spend more
time on it.**

### Three errors in the research above, caught before they cost anything
Consistent with `learning.md` "verify subagent research findings before acting":

1. `huihui_ai/Qwen3.6-abliterated` **has no `latest` tag** — the pull command in
   this issue fails with `pull model manifest: file does not exist`. The "HTTP
   200 verified" check hit the model *page*, not a pullable manifest.
2. Its "17 GB" is wrong: every `35b*` tag is **23.94 GB**. On this 32 GB machine
   (default `iogpu.wired_limit_mb=0` → ~24 GiB) that spills past the GPU wired
   limit once KV cache and the MLX audio models are resident — it would have been
   **slower**, not faster.
3. The 17 GB figure matches the `27b` tag (17.42 GB), which is **dense** (no
   `-a3b`) and so defeats the entire premise.

### Where this now points
Test the same model family through the **MLX runtime** —
`mlx-community/gemma-4-26B-A4B-it-heretic-4bit` (15.6 GB). `mlx_lm` 0.31.3 is
already installed via `mlx-audio`, so this needs no dependency change.
`scripts/bench_llm_mlx.py` measures it with metrics identical to `bench_stack.py`.

Caveat for whoever picks this up: if MLX wins, the fix is a **runtime change**,
not the "one env var" swap step 5 promises. That is a bigger blast radius than
this issue was scoped for and needs Sid's call before touching `voice_agent.py`.

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
