---
id: 0001
title: Barge-in ignores overlapping speech, and misfires at end of replies
status: open
priority: urgent
area: audio/aec
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Two failures in the same mechanism:

1. **Real interruptions are ignored.** Speaking while the agent talks does not stop
   it, and the overlapping speech is never recorded. The agent only listens once it
   has finished its reply. Sid: *"if there is an overlapping talk I have, it's
   ignored, only after it finishes it listens to me — that's not how normal
   conversations go."*
2. **False barge-ins still fire**, and specifically near the *end* of a reply.

## Why it matters

Full-duplex interruption is the difference between a voice agent and a walkie-talkie.
This is the single largest remaining gap in how natural the agent feels.

## Evidence

Session 2026-07-30 (`TURN_DETECTOR=off`, built-in mic + speakers), 3 barge-ins across
10 turns, **every one reporting `gate=0.050`** — the fixed floor, never the adaptive
value:

```
AI: Hey! You're back already? Or just checking in? [barge-in] cutting reply (rms=0.128 gate=0.050 streak=6)
AI: Right? It's like she's just trying to jump in ... [barge-in] cutting reply (rms=0.141 gate=0.050 streak=6)
```

The adaptive gate (`max(floor, BARGE_IN_ECHO_FACTOR * player.output_rms())`) is
therefore not engaging when it fires. Likely cause: `_note_output`'s decay
(alpha 0.05) drives `output_rms` toward zero through trailing silence at the end of a
sentence, while `state["tts_audible"]` is still True — so the gate collapses to the
floor exactly where residual echo and room noise sit.

Meanwhile the conjunction added to stop self-interruption
(`rms >= gate and is_speech`, sustain 6 frames) is evidently too strict for genuine
double-talk: the user's voice arrives *mixed with* the agent's output, so AEC
residual plus attenuation keeps it under the bar.

Prior art from research (2026-07-30): a practitioner with a near-identical Mac stack
reported *"AEC reduces both the echo AND your voice to ~0.001 RMS... pVAD gives
0.82-0.94 on Kokoro's TTS echo, barely different from real speech. All failed on
laptop speakers. Disabled barge-in for now."* Pure-software AEC on built-in MacBook
geometry may not be solvable by threshold tuning alone.

## Approach

Do not tune thresholds further — the two failure modes pull in opposite directions
and there is no single RMS bar that satisfies both. Options, roughly in order:

1. **Decide on a real signal, not energy.** Feed the AEC-cleaned mic to a
   speech/non-speech model continuously during TTS and require sustained *speech
   probability*, not amplitude. Cheap enough (Silero is ~1 ms/frame).
2. **Hold the gate up for the whole reply** rather than letting `output_rms` decay
   between sentences — decay should track "reply finished", not "this frame is quiet".
3. **Test with headphones first** to separate "barge-in logic is wrong" from "AEC
   cannot recover the user's voice on this hardware". If it works on headphones, the
   logic is fine and the built-in-speaker case may need `HALF_DUPLEX=1` honesty
   instead of a broken promise.
4. Evaluate against **recorded double-talk**, not live sessions — see [[0005]].

Relevant knobs: `BARGE_IN_RMS_GATE`, `BARGE_IN_ECHO_FACTOR`,
`BARGE_IN_SUSTAIN_FRAMES`, `STREAM_DELAY_MS`.
