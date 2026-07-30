# Learning

Durable lessons for this project. Newest first. Append before continuing work — this is
how the project gets smarter rather than repeating itself.

---

## 2026-07-30 · Check the premise of a plan before executing it, not just the steps

**Context:** `issues/0003` said "the LLM is slow because it's dense — swap in a sparse
3B-active MoE, expect ~3.7x". The plan was detailed, evidenced, and marked urgent. The
model we were already running turned out to *be* a sparse MoE (`expert_count = 128`,
`expert_used_count = 8`, 100% GPU) that simply decodes at dense speed. The whole issue
was "replace X with Y" where we already had Y. A control run confirmed it: a second
same-class GGUF landed at 13.0 tok/s vs 12.5 — inside the noise.

**Rule:** Before executing a written plan — including one this project wrote itself —
spend the five minutes to verify its central factual claim against the artifact in
front of you. A plan's steps get scrutiny because they're what you're about to type;
its premise sits above them and gets read as given. Ask "is the thing this says is
true, actually true here?" `ollama show` / model metadata / `ollama ps` answer it
faster than any benchmark.

**Example:** Three claims in 0003 were also false — the recommended tag has no
`latest` and cannot be pulled at all, its real size is 23.94 GB not 17 GB (on this
32 GB machine that exceeds the ~24 GiB GPU wired limit and would have run *slower*),
and the 17 GB figure belonged to a dense variant. This is the same failure the
"verify subagent research findings" lesson below describes, surviving into a written,
prioritised issue — so the check has to happen at execution time too, not only when
the research lands.

---

## 2026-07-30 · A faster number in one column can still be a slower agent

**Context:** MLX decoded 20% faster than llama.cpp (18.3 vs 15.2 tok/s) — the metric
0003 was explicitly optimising. It was still the wrong choice: its prompt *prefill*
ran at 51–60 tok/s. With `HISTORY_MAX_TURNS=8` the prompt grows every turn, so the
agent would have got slower the longer you talked to it. The win column was real and
irrelevant.

**Rule:** When benchmarking a swap, measure every stage the change touches, including
the ones the current setup is silently good at. A regression hides in the column you
didn't think to print — especially costs that scale with conversation length, since a
fresh-prompt benchmark never sees them.

**Example:** `bench_llm_mlx.py` originally reported TTFT / first-sentence / decode. It
took a separate probe with a 17-token prompt to expose `prompt_tps`, which is what
actually killed the option.

---

## 2026-07-30 · Component tests cannot catch conversation bugs

**Context:** Three changes shipped clean through every scripted test and broke immediately
in live use — utterance fragmentation (buffer wiped on resume), a fatal GIL crash on the
second turn, and runaway 20-second STT buffers. Sid found all three by talking to the
agent, across five test sessions.

**Rule:** A voice agent's bugs live in the *interaction* between stages and threads, not
in the stages. Before shipping anything that touches turn-taking, buffering, or
threading, it must be exercised by (a) **multiple turns** and (b) **a worker thread** —
`scripts/smoke_multiturn.py` exists for exactly this. `smoke_pipeline.py` runs one turn
on the main thread and is structurally incapable of catching this class of bug.

**Example:** `voice_agent.py` spawned a fresh thread per turn. MLX keeps thread-local
Metal state, and destroying it on thread exit crashes the interpreter
(`PyThreadState_Get: ... the GIL is released`). Kokoro alone tolerated it; adding a
second MLX model made it fatal on turn 2. Fix: one persistent worker that never exits.

---

## 2026-07-30 · Never bound one dimension of an unbounded wait

**Context:** Semantic turn detection added a "keep listening if the thought isn't
finished" path. Fixing its first bug (buffer reset) removed a ceiling without adding one,
producing an 11.5 s transcription. Adding a *duration* cap still left the loop appending
silence while it waited, so every turn simply grew until it hit the cap — 18–21 s STT.

**Rule:** When adding a wait-for-more-input path, bound **every** dimension at once:
iteration count, total duration, and what accumulates during the wait. Bounding one at a
time produces a new pathology per fix.

**Example:** `MAX_UTTERANCE_S` alone was insufficient; it needed `MAX_TURN_CONTINUATIONS`
too, and the buffer still accumulates silence frames (unresolved — see `issues/0005`).

---

## 2026-07-30 · Measure on this machine; published latency numbers are near-useless

**Context:** Every latency figure found in research came from an M4 Max, an H100, or a
vendor blog. The project's own `CLAUDE.md` carried numbers that were wrong by 4–5×.

**Rule:** Before choosing a model, measure it here. `scripts/bench_stack.py` (per-slot)
and `scripts/compare_stt.py` (real speech, WER-scored) exist for this. Also verify the
*right* metric: for a per-sentence TTS pipeline the gate is **time-to-first-sentence**,
not TTFT — they differed 3× on the same model.

**Example:** Documented "STT ~75 ms" measured 351–396 ms. "Kokoro 17× realtime" measured
8.9×. Whisper's claimed accuracy crown lost to Parakeet on Sid's actual voice.

---

## 2026-07-30 · Verify subagent research findings before acting on them

**Context:** A research subagent's top-ranked STT recommendation (`nemotron-asr-mlx`) was
a **1-star repo, last pushed 5 months earlier**, wrapping an English-only model while the
agent claimed it gave 40 language-locales. Another agent reported a WER table that a web
AI summary had printed backwards.

**Rule:** Treat subagent output as leads, not conclusions. Check repo health (stars,
last push, release count), confirm the model behind a wrapper is the one claimed, and
prefer primary sources (HF model cards, GitHub releases, the project's own benchmark
files) over blog summaries. Programmatic-SEO benchmark sites read authoritative and are
not.

**Example:** `mlx-audio` (⭐7650, pushed daily) turned out to already contain the
Nemotron MLX port the agent said had "no Apple Silicon path".

---

## 2026-07-30 · A comment that describes a condition the code doesn't implement

**Context:** Barge-in fired on the agent's own voice, cutting 7 of 13 replies. The
condition's comment read *"if user has been loud for N consecutive frames AND VAD
currently agrees there's speech"* — the code checked only energy. The VAD term had never
existed.

**Rule:** When debugging, read the condition, not its comment. And when a comment
describes a safeguard, grep for the safeguard.
