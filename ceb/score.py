"""Turn raw runs into the numbers that go on the leaderboard.

Two metrics here are the reason the project exists and almost never appear in
portfolio work:

  hallucination_rate - of the cases where the criteria text does not state a
      value, how often did the model invent one? This is the failure that
      matters in production and it is invisible to plain accuracy.

  json_validity - how often the output parsed and satisfied the schema at all.
      A model with better accuracy on parseable rows and worse validity can
      easily be the worse model.

Accuracy is reported per field, never as one headline number, because the
fields fail at very different rates and averaging hides it.
"""

import argparse
import json
import math
import pathlib
from collections import defaultdict

from .age import close
from .schema import FIELDS

RAW = pathlib.Path("results/raw.jsonl")
OUT = pathlib.Path("results/scores.json")


def wilson(successes: int, total: int, z: float = 1.96):
    """95% interval. With 60 examples the interval is wide, and saying so is
    the difference between a benchmark and a number."""
    if total == 0:
        return [0.0, 0.0]
    p = successes / total
    denom = 1 + z ** 2 / total
    centre = (p + z ** 2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z ** 2 / (4 * total ** 2)) / denom
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


def field_match(field, gold, pred):
    if field in ("min_age_years", "max_age_years"):
        return close(gold, pred)
    return gold == pred


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return round(ordered[idx], 3)


def score(raw_path=RAW, out_path=OUT):
    rows = [json.loads(l) for l in
            raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["prompt"])].append(row)

    results = []
    for (model, prompt), group in sorted(groups.items()):
        n = len(group)
        parsed = [r for r in group if r["pred"] is not None]
        violations = defaultdict(int)
        for r in group:
            if r["violation"]:
                violations[r["violation"].split(":")[0]] += 1

        per_field, hallucinated, missed, stated_total, silent_total = {}, 0, 0, 0, 0
        for field in FIELDS:
            hits = 0
            for r in parsed:
                gold, pred = r["gold"][field], r["pred"][field]
                if field_match(field, gold, pred):
                    hits += 1
                if gold is None and pred is not None:
                    hallucinated += 1
                if gold is not None and pred is None:
                    missed += 1
                if gold is None:
                    silent_total += 1
                else:
                    stated_total += 1
            per_field[field] = {
                "accuracy": round(hits / len(parsed), 4) if parsed else 0.0,
                "ci95": wilson(hits, len(parsed)),
                "n": len(parsed),
            }

        latencies = [r["latency_s"] for r in group if not r["transport_error"]]
        completion = sum(r["completion_tokens"] for r in group)

        results.append({
            "model": model,
            "prompt": prompt,
            "n_examples": n,
            "json_validity": round(len(parsed) / n, 4) if n else 0.0,
            "json_validity_ci95": wilson(len(parsed), n),
            "violations": dict(violations),
            "per_field": per_field,
            "mean_field_accuracy": round(
                sum(v["accuracy"] for v in per_field.values()) / len(FIELDS), 4),
            "hallucination_rate": round(hallucinated / silent_total, 4) if silent_total else None,
            "miss_rate": round(missed / stated_total, 4) if stated_total else None,
            "latency_p50_s": percentile(latencies, 50),
            "latency_p95_s": percentile(latencies, 95),
            "completion_tokens_total": completion,
            "transport_errors": sum(1 for r in group if r["transport_error"]),
            "usd_per_1k_docs": 0.0,  # local models; set by the hosted adapter
        })

    results.sort(key=lambda r: (-r["mean_field_accuracy"], r["latency_p50_s"] or 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"scored {len(rows)} rows across {len(results)} model/prompt pairs")
    for r in results[:6]:
        print(f"  {r['model']:<26} {r['prompt']:<12} "
              f"acc={r['mean_field_accuracy']:.3f}  "
              f"json={r['json_validity']:.3f}  "
              f"halluc={r['hallucination_rate']}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=pathlib.Path, default=RAW)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    score(args.raw, args.out)


if __name__ == "__main__":
    main()
