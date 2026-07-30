# uncensored-local-voice — State

_Last updated: 2026-07-30_

## Now
- Branch **`voice-stack-upgrade-2026-07`** (12 commits, all pushed, worktree clean). `main` untouched.
- **Known-good config is `TURN_DETECTOR=off`** — Sid confirmed it works. Run it that way.
- Stack: **Parakeet TDT v3** STT · Kokoro TTS · Silero VAD (min_silence 500 ms with detector
  off) · WebRTC AEC · SuperGemma4 LLM via Ollama. Unchanged this session — **no runtime code
  was touched**, only docs, the issue tracker, and a new bench script.
- `issues/` has **8 open** (3 urgent). `0003` is measured out and should be **set aside**.
- Last verified: `engines.py` imports and still builds the Parakeet default; the LLM A/B ran
  clean across three configurations (rows in `bench_results.jsonl`).

## Next
1. **`issues/0002` — inter-sentence gap.** Now the top item, since 0003 is spent. Callback-driven
   output stream + ring buffer so TTS generates ahead of playback. Must move the AEC
   reverse-stream feed carefully.
2. **`issues/0001` — barge-in.** Do after 0002; they share the playback mechanism.
3. `issues/0004` (STT slower in-pipeline than isolated — may be free latency), then 0007 / 0005 / 0006 / 0008.
4. Optional, Sid deferred it ("later"): upgrade Ollama 0.21.2 → 0.32.5 for its own MLX engine.
   Only remaining LLM lever; see the ceiling estimate below before spending on it.

## Blockers
- None. The one open decision — whether to upgrade Ollama — Sid deferred.

## Latest handoff — 2026-07-30 (evening: issue 0003 measured to a dead end, docs corrected)

### What shipped
Three commits, all pushed. **No change to `voice_agent.py` — the agent behaves exactly as it
did at `ddb6369`.**

- `5ea6a74` — `CLAUDE.md` corrected to the shipped stack; `engines.py` docstring fixed; new
  `scripts/bench_llm_mlx.py`.
- `dbf9b17` — `issues/0003` rewritten: the premise was wrong.
- `1c77efe` — `issues/0003` final: MLX measured, both levers dead.

### The finding: issue 0003's premise was false, and both fixes it implied are dead
0003 said the LLM is slow because it is dense, so swap in a sparse 3B-active MoE. **The model
we already run is a sparse MoE** — `gemma4.expert_count = 128`, `expert_used_count = 8`, at
`100% GPU`, 19 GB — and still decodes at dense speed. So the premise did not hold, and the two
candidate fixes both measured out:

| | decode | TTFT | prefill |
|---|---|---|---|
| **current GGUF / llama.cpp** | 15.2 tok/s | **276 ms** | fast |
| second GGUF (`HammerAI/gemma-4-26b-a4b-heretic`) | 13.0–16.6 tok/s | 266 ms | fast |
| MLX / mlx-lm (`mlx-community/gemma-4-26B-A4B-it-heretic-4bit`) | **18.3 tok/s** | 529–1077 ms | **51–60 tok/s** |

- **GGUF→GGUF: no change.** Inside the noise. Dead.
- **MLX: +20% decode, but prefill is disqualifying.** With `HISTORY_MAX_TURNS=8` the prompt
  grows every turn, so a ~55 tok/s prefill means the agent gets *slower the longer you talk to
  it*. Confirmed warm, 17-token prompt, 3 trials — not cold start.
- **That MLX model is separately unusable:** it opens a `<|channel>thought` block on every
  generation, even for a bare "Hi" with no system prompt, despite its template correctly
  pre-filling an empty closed thought block when thinking is off. mlx-lm has no `think: false`.

### Three errors in 0003's original research, caught before they cost anything
1. `huihui_ai/Qwen3.6-abliterated` **has no `latest` tag** — the pull command in the issue
   fails outright. The "verified HTTP 200" check hit the model page, not a manifest.
2. Its "17 GB" is wrong: every `35b*` tag is **23.94 GB**. On this 32 GB machine (default
   `iogpu.wired_limit_mb=0` → ~24 GiB) that spills past the GPU wired limit — it would have run
   **slower**, not faster.
3. The 17 GB figure belongs to the `27b` tag, which is **dense** and defeats the premise.

### What to do first next session
Start `issues/0002`. **Do not reopen 0003** — read its "Verdict" section first; the numbers and
both dead ends are written up there so nobody re-runs this. The only untested lever is the
Ollama 0.21.2 → 0.32.5 upgrade (its bundled MLX engine has custom small-batch matmul kernels and
a v0.31.1 fix for "Gemma 4 MoE model loading", so it may not repeat mlx-lm's prefill weakness) —
but the measured ceiling is ~20% of one stage in exchange for replacing the engine the whole
agent depends on. Sid said "later". 0002 and 0001 are smaller but certain, and they are what
make the agent *feel* natural.

### Gotchas discovered this session (they will bite again)
- **`ollama stop <model>` before any MLX benchmark**, or it OOMs the GPU at 19 GB resident
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`).
- **`apply_chat_template(tokenize=False)` → `stream_generate(str)` double-prepends BOS.** Pass
  token ids. The first MLX run was invalid because of this.
- **`ollama pull` can stall silently at byte-complete** — it sat 25 min with the client at 0%
  CPU and the blob still `-partial`. Kill and re-run; blobs resume.
- Benchmarks run while a large download is in flight read **~20% low**. Re-run clean before
  believing any A/B.
