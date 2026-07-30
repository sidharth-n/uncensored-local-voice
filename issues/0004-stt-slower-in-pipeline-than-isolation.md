---
id: 0004
title: STT is 5-10x slower in the live pipeline than in isolation
status: open
priority: moderate
area: stt/threading
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Parakeet transcribes far slower inside the running agent than the same model on the
same audio in a standalone script.

## Why it matters

~700 ms of unexplained per-turn latency. If it is contention, fixing it is free
speed; if it is real, the STT benchmark numbers used to pick the default are
misleading.

## Evidence

Isolated (`compare_stt.py` / direct loop, 5 runs, median):

```
 4s audio ->  76 ms
 8s audio -> 104 ms
20s audio -> 276 ms
45s audio -> 977 ms
```

Live session 2026-07-30, with audio length now logged:

```
[stt 1004ms on 1.1s]
[stt  767ms on 7.0s]
[stt  710ms on 4.5s]
[stt 2569ms on 1.0s]
[stt 1177ms on 3.2s]
```

1.1 s of audio taking 1004 ms is ~13x the isolated figure for that length, and the
cost does not track length — which points at contention rather than workload.

## Approach

Hypotheses, cheapest first:

1. **Main-thread/worker contention.** The main loop runs Silero VAD (torch) on every
   32 ms frame while the worker runs Parakeet (MLX). Both compete for GPU and the GIL.
   Test by timing STT with the mic loop idle vs busy.
2. **Cold Metal kernels per call.** The isolated test warms up once then loops
   immediately; live calls are seconds apart. Check whether a periodic keep-warm
   inference restores the fast path.
3. **Temp-wav overhead.** `MlxAudioStt.transcribe` writes a wav per call because
   mlx-audio's `generate()` takes a path. Should be microseconds, but confirm.
4. Consider whether `mlx-audio` re-runs setup per `generate()` call.

Note the isolated numbers were what justified making Parakeet the default over
Moonshine (396 ms). Even at live cost, Parakeet still wins — Moonshine measured
812-1626 ms in the same live conditions — so the decision stands, but the margin is
smaller than the benchmark implied.
