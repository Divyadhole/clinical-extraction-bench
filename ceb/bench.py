"""Run every model x prompt over the corpus. Resumable, deterministic."""

import argparse
import json
import pathlib

from . import runners
from .prompts import PROMPTS, render
from .schema import parse

RAW = pathlib.Path("results/raw.jsonl")


def load_done(path):
    done = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done.add((row["model"], row["prompt"], row["nct_id"]))
    return done


def run(models, prompts, corpus_path, raw_path=RAW, limit=None):
    corpus = [json.loads(l) for l in
              corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        corpus = corpus[:limit]

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(raw_path)
    total = len(models) * len(prompts) * len(corpus)
    completed = 0

    with raw_path.open("a", encoding="utf-8") as fh:
        for spec in models:
            runner = runners.build(spec)
            for prompt_name in prompts:
                for example in corpus:
                    completed += 1
                    key = (spec, prompt_name, example["nct_id"])
                    if key in done:
                        continue

                    result = runner.generate(
                        render(prompt_name, example["criteria"]))
                    extraction, violation = parse(result.text)

                    fh.write(json.dumps({
                        "model": spec,
                        "prompt": prompt_name,
                        "nct_id": example["nct_id"],
                        "gold": example["gold"],
                        "pred": extraction.model_dump() if extraction else None,
                        "violation": violation,
                        "transport_error": result.error,
                        "latency_s": round(result.latency_s, 4),
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                    }, ensure_ascii=False) + "\n")
                    fh.flush()

                    if completed % 10 == 0:
                        print(f"  {completed}/{total}  {spec} / {prompt_name}")

    print(f"done: {completed} cells, results in {raw_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="e.g. ollama:qwen2.5:3b ollama:llama3.2:3b")
    ap.add_argument("--prompts", nargs="+", default=sorted(PROMPTS))
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("data/corpus.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.models, args.prompts, args.corpus, limit=args.limit)


if __name__ == "__main__":
    main()
