---
id: 0008
title: Fatal GIL error printed on Ctrl+C shutdown (cosmetic)
status: open
priority: low
area: threading
opened: 2026-07-30
updated: 2026-07-30
closed:
---

## What

Quitting with Ctrl+C can print a `Fatal Python error: PyThreadState_Get` block *after*
`bye 👋`, plus a leaked-semaphore warning.

## Why it matters

Purely cosmetic — it happens after the conversation has ended and nothing is lost. It
is noted only so it is not mistaken for the mid-conversation crash that was fixed in
b8be6e1, which was real and fatal.

## Evidence

Same signature as the fixed bug: MLX keeps thread-local Metal state, and destroying it
when a thread exits calls back into Python without the GIL. The mid-conversation case
was fixed by moving from a thread-per-turn to one persistent worker; the worker is a
daemon, so it is still torn down abruptly at interpreter shutdown.

```
bye 👋
Fatal Python error: PyThreadState_Get: the function must be called with the GIL held...
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

## Approach

Signal the worker to finish and exit cleanly before the interpreter tears down — or,
more bluntly, `os._exit(0)` after `save_history()` so no MLX-touching thread is ever
destructed. The second is ugly but appropriate for a process whose job is finished, and
avoids depending on MLX's teardown behaviour.

Low priority; do not bundle with audio changes.
