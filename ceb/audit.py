"""Measure how good the free ground truth actually is.

The registry's structured fields are not always supported by the criteria
prose. A benchmark built on found labels has to say how often its labels are
wrong, or its ceiling is unknown and every number above is uninterpretable.

Shows a sample of trials one at a time and asks, per field, whether the prose
actually states the registry's value. Twenty minutes for 40 trials.
"""

import argparse
import json
import pathlib
import random

from .schema import FIELDS

OUT = pathlib.Path("data/audit.jsonl")


def run(corpus_path, out_path=OUT, n=40, seed=7):
    corpus = [json.loads(l) for l in
              corpus_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(seed).shuffle(corpus)
    already = set()
    if out_path.exists():
        already = {json.loads(l)["nct_id"]
                   for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    todo = [e for e in corpus if e["nct_id"] not in already][:n - len(already)]
    if not todo:
        print("audit already complete")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("For each field: y = the prose supports this value, n = it does not, "
          "s = skip. Ctrl-C saves and exits.\n")

    with out_path.open("a", encoding="utf-8") as fh:
        for i, example in enumerate(todo, 1):
            print("=" * 78)
            print(f"[{i}/{len(todo)}] {example['nct_id']}  {example['title']}")
            print("-" * 78)
            print(example["criteria"][:1800])
            print("-" * 78)
            verdicts = {}
            try:
                for field in FIELDS:
                    gold = example["gold"][field]
                    answer = ""
                    while answer not in {"y", "n", "s"}:
                        answer = input(f"  {field} = {gold!r}  supported? [y/n/s] ").strip().lower()
                    verdicts[field] = answer
            except (KeyboardInterrupt, EOFError):
                print("\nstopping, progress saved")
                break
            fh.write(json.dumps({"nct_id": example["nct_id"], "verdicts": verdicts}) + "\n")
            fh.flush()

    summarise(out_path)


def summarise(out_path=OUT):
    rows = [json.loads(l) for l in
            out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return
    print(f"\nLabel quality over {len(rows)} audited trials:")
    for field in FIELDS:
        judged = [r["verdicts"][field] for r in rows if r["verdicts"].get(field) in "yn"]
        if judged:
            rate = judged.count("y") / len(judged)
            print(f"  {field:<28} {rate:.1%} of registry values supported by the prose")
    print("\nQuote these in the README. They are the benchmark's ceiling.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=pathlib.Path,
                    default=pathlib.Path("data/corpus.jsonl"))
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()
    if args.summarise_only:
        summarise()
    else:
        run(args.corpus, n=args.n)


if __name__ == "__main__":
    main()
