---
id: 0005
title: Semantic turn detection parked — needs offline evaluation, not live debugging
status: open
priority: moderate
area: vad/turn-taking
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

`TURN_DETECTOR=smartturn` is implemented and works in isolation, but caused three
separate regressions in live use and is **not** the default. `TURN_DETECTOR=off` is
the known-good configuration.

## Why it matters

The idea is still right: a fixed silence timeout cannot tell a pause from a finished
thought, so it both delays every turn and cuts people off. But the feature cost five
live test sessions and it should not be re-enabled until it can be evaluated without
Sid's time.

## Evidence

Isolated, the model is good — `scripts/smoke_turn.py` on this M5:

```
latency   19.1 ms median (18.8-20.2)   vendor claims "<60 ms CPU"
accuracy  8/8 at threshold 0.6, 7/8 at the stock 0.5
```

Wide separation: complete utterances 0.834-0.957, incomplete 0.017-0.330.

In the pipeline it caused, in order:

1. **Utterance fragmentation.** The "keep listening" path kept `active=True`, then the
   next Silero `start` event unconditionally reset `speech_buf`, discarding everything
   before the pause. Transcripts became tail fragments — "No.", "Thank you." Fixed in
   de96969.
2. **Runaway buffers.** With the reset removed there was no ceiling, and every frame
   (silence included) kept accumulating while the detector deferred. One turn showed
   `stt 11541ms`; a later session showed 18-21 s repeatedly. Partially fixed in
   0df0154 (`MAX_UTTERANCE_S`, `MAX_TURN_CONTINUATIONS`).
3. **Interaction with barge-in.** Lowering `min_silence` 500 -> 200 ms made pauses far
   more frequent, multiplying the frequency of both bugs above.

Root process failure: every one of these appeared only in multi-turn live conversation.
`smoke_pipeline.py` runs one turn on the main thread; `smoke_turn.py` scores the model
on isolated clips. Neither could catch an interaction bug.

## Approach

Do not re-enable by shipping to Sid and reading logs afterwards.

1. **Record real conversations first.** Capture raw mic audio plus VAD events to disk
   during normal `TURN_DETECTOR=off` use, so there is a corpus of real pauses,
   hesitations and completions.
2. **Replay offline.** Build a harness that runs the turn detector over recorded
   sessions and reports: how often it deferred, how long each utterance grew, how many
   times the caps fired, and whether the resulting segmentation matches where Sid
   actually finished speaking.
3. Only after that produces sensible numbers, enable it live.
4. Malayalam: not in Smart Turn's 23 languages, and it reads lexical content rather
   than pure prosody, so it should not be assumed to generalize. Gate by language if
   Malayalam is revived ([[0006]]).

Keep `MAX_UTTERANCE_S` and `MAX_TURN_CONTINUATIONS` regardless — they are cheap
insurance against any future unbounded wait.
