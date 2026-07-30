# Learning

Durable lessons for this project. Newest first. Append before continuing work — this is
how the project gets smarter rather than repeating itself.

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
