---
id: 0002
title: Audible gap before the second sentence (TTS generates synchronously)
status: open
priority: urgent
area: tts/playback
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Speech stalls between sentences. The first sentence plays, then there is a pause, then
the second begins. Sid: *"generating the second sentence after one"*.

## Why it matters

Human speech does not pause 200-300 ms between every sentence. This reads as the
machine thinking rather than talking, and it is one of the two things Sid named as
making the agent feel unnatural.

## Evidence

`TTSPlayer.speak_sentence()` generates **and** plays in one synchronous pass, and
`respond()` calls it per sentence in a loop:

```python
for sent in sentence_stream(stream_llm(...)):
    player.speak_sentence(sent)     # generates AND blocks until played out
```

So sentence N+1's synthesis only starts after sentence N has finished playing.
Kokoro measured at 14.7x realtime, so a ~3 s sentence costs ~200 ms to generate —
which becomes dead air, every sentence boundary.

## Approach

Decouple generation from playback:

- Replace the blocking `OutputStream.write()` with a **callback-driven output stream
  fed from a ring buffer**, so playback drains audio while the worker generates ahead.
- Generation must stay on the single worker thread — MLX thread-local Metal state
  makes a second MLX thread unsafe (see commit b8be6e1: thread teardown crashes the
  interpreter). The *playback* callback does no MLX, so it is safe.
- **The AEC reverse stream must move with playback.** `process_reverse_stream()` has
  to be fed the frames the speaker actually emits, in 10 ms frames at 16 kHz, or echo
  cancellation breaks. This is the risky part and the reason it was not bundled with
  other fixes.
- Prefetch depth of one sentence is probably enough; verify against `bench_stack.py`
  TTFA and by ear.

Do not attempt alongside other changes — this touches the mechanism [[0001]] depends
on.
