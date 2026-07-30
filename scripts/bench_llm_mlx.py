"""Benchmark an MLX-format LLM with the same metrics as bench_stack.py --slot llm.

Why this exists: the agent's model is a GGUF, so Ollama runs it on llama.cpp even
though this machine's Ollama build ships an MLX runner. The current model is
already a 4B-active MoE (128 experts, 8 used) and still decodes at ~14-16 tok/s,
which is dense-model speed — so the suspicion is the runtime, not the
architecture. This measures the same model class through MLX to test that.

Metrics match bench_stack.bench_llm exactly so rows are comparable in
bench_results.jsonl:
  - TTFT              first token out
  - first sentence    what actually gates audio (TTS is driven per sentence)
  - decode tok/s      the rate first-sentence latency is a function of

Usage
-----
    uv run python scripts/bench_llm_mlx.py mlx-community/gemma-4-26B-A4B-it-heretic-4bit
    uv run python scripts/bench_llm_mlx.py <repo> --repeat 3 --text "..."
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from voice_agent import SYSTEM_PROMPT, sentence_stream  # noqa: E402
from bench_stack import PROMPTS, record, summarize  # noqa: E402


def bench(repo: str, prompt: str, repeat: int, lang: str, max_tokens: int) -> dict:
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    t0 = time.perf_counter()
    model, tokenizer = load(repo)
    load_ms = (time.perf_counter() - t0) * 1000

    # Match the Ollama model's sampling params so the comparison isn't confounded
    # by a different sampler doing more or less work per token.
    sampler = make_sampler(temp=1.0, top_p=0.95, top_k=64)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    # Tokenize here rather than passing a string: apply_chat_template emits
    # bos_token, and stream_generate would tokenize the string again and prepend
    # a second BOS. The doubled BOS made the model ignore the template and open
    # a <|channel>thought block, which is exactly what `think: false` avoids on
    # the Ollama side — and it would have made the two runs incomparable.
    formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    ttft, first_sentence, tok_s, replies = [], [], [], []

    for i in range(repeat + 1):  # first iteration warms; discarded, as in bench_stack
        t0 = time.perf_counter()
        first = None
        sent_at = None
        text: list[str] = []
        last_tps = None

        def tokens():
            nonlocal first, last_tps
            for resp in stream_generate(
                model, tokenizer, formatted, max_tokens=max_tokens, sampler=sampler
            ):
                chunk = resp.text
                if chunk and first is None:
                    first = (time.perf_counter() - t0) * 1000
                if chunk:
                    text.append(chunk)
                    yield chunk
                last_tps = resp.generation_tps

        # Same splitter the live agent uses, so first-sentence is the real one.
        for _ in sentence_stream(tokens()):
            if sent_at is None:
                sent_at = (time.perf_counter() - t0) * 1000

        if i == 0:
            continue
        if first is not None:
            ttft.append(first)
        if sent_at is not None:
            first_sentence.append(sent_at)
        if last_tps:
            tok_s.append(last_tps)
        replies.append("".join(text).strip())

    if not ttft:
        raise SystemExit(f"{repo} produced no content")

    row = {
        "slot": "llm",
        "engine": repo,
        "runtime": "mlx-lm",
        "lang": lang,
        "load_ms": round(load_ms, 1),
        "ttft": summarize(ttft),
        "first_sentence": summarize(first_sentence) if first_sentence else None,
        "tokens_per_s": round(statistics.median(tok_s), 1) if tok_s else None,
        "reply": replies[-1],
    }
    record(row)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HF repo id of an MLX-format model")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--lang", default="en", choices=sorted(PROMPTS))
    ap.add_argument("--text", default=None)
    ap.add_argument("--max-tokens", type=int, default=120)
    args = ap.parse_args()

    prompt = args.text or PROMPTS[args.lang]
    print(f"=== bench: mlx-lm lang={args.lang} repeat={args.repeat}")
    print(f"    prompt: {prompt!r}\n")

    r = bench(args.repo, prompt, args.repeat, args.lang, args.max_tokens)
    fs = r["first_sentence"]
    print(f"[llm/mlx] {r['engine']}  (load {r['load_ms']} ms)")
    print(f"      TTFT           {r['ttft']['median_ms']} ms median "
          f"(min {r['ttft']['min_ms']}, max {r['ttft']['max_ms']})")
    print(f"      first sentence {fs['median_ms'] if fs else '?'} ms median "
          f"<- this is what gates audio")
    print(f"      decode         {r['tokens_per_s']} tok/s")
    print(f"      -> {r['reply'][:160]!r}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
