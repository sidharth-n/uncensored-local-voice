# Issues

Local issue tracker. One file per issue: `NNNN-slug.md`, frontmatter tags
(`id`, `title`, `status`, `priority`, `area`, `opened`, `updated`, `closed`).
New issues start from `_template.md`.

**Open: 8** — Urgent 3 · Moderate 3 · Low 2

## Urgent

| id | title | area |
|---|---|---|
| [0001](0001-barge-in-ignores-overlapping-speech.md) | Barge-in ignores overlapping speech, and misfires at end of replies | audio/aec |
| [0002](0002-gap-before-second-sentence.md) | Audible gap before the second sentence (TTS generates synchronously) | tts/playback |
| [0003](0003-llm-decode-rate-dominates-latency.md) | LLM decode rate dominates latency — swap to a 3B-active MoE | llm |

## Moderate

| id | title | area |
|---|---|---|
| [0004](0004-stt-slower-in-pipeline-than-isolation.md) | STT is 5-10x slower in the live pipeline than in isolation | stt/threading |
| [0005](0005-semantic-turn-detection-parked.md) | Semantic turn detection parked — needs offline evaluation, not live debugging | vad/turn-taking |
| [0007](0007-proper-nouns-need-decoder-biasing.md) | Proper nouns unrecognisable by every STT engine — needs biasing | stt |

## Low

| id | title | area |
|---|---|---|
| [0006](0006-malayalam-support-parked.md) | Malayalam support parked until the English path is solid | multilingual |
| [0008](0008-fatal-gil-error-on-shutdown.md) | Fatal GIL error printed on Ctrl+C shutdown (cosmetic) | threading |

## Suggested order

`0003` first — it is a one-line, reversible model swap worth ~1 s off every reply, and
it does not touch the audio path. Then `0002`, then `0001`; those two share the playback
mechanism and `0002` should land first because `0001` depends on it. `0004` is cheap to
investigate and may be free latency. `0005`/`0006` are deliberately deferred.
