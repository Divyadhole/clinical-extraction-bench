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

TWO ACCURACY DENOMINATORS
-------------------------
The first version of this file reported only one accuracy: hits divided by the
number of rows that *parsed*. That flatters any model that copes with a hard
document by emitting nothing at all, because the row it failed on silently
leaves the denominator. The first real run made the problem concrete:
llama3.2:3b / zero_shot scored 0.425 on parsed rows while parsing only 33% of
them, and sorted third on a leaderboard where it belongs last.

So both are reported now:

  accuracy (per field)    hits / parsed rows. Answers "when it answers, is it
                          right?" - the right question for prompt debugging.
  end_to_end_accuracy     hits / (rows x fields), an unparsed row scoring zero
                          on every field. Answers "if I run this over 60
                          documents, what fraction of the values do I get?" -
                          the only one of the two you can plan around.

The leaderboard sorts on end_to_end_accuracy. Its interval is clustered by
trial rather than by field, because the four fields inside one trial fail
together often enough that treating them as 4N independent draws would claim
about twice the precision the data supports.

BASELINES
---------
Two rows on the leaderboard never call a model:

  baseline:majority   always answers the most common value for each field
                      (18 years, no maximum, ALL, does not accept healthy
                      volunteers). It reads nothing.
  baseline:all-null   always answers null. It cannot hallucinate by
                      construction, which is the point of including it.

The majority baseline picks its constants from the same 60 examples it is
scored on, so it is an optimistic version of a trivial system. The honest
caveat is mild here: those four constants are also what anyone who has read
ten trial registrations would guess without looking at this corpus at all.

A benchmark that omits these rows can report a model at 43% and let the reader
assume 43% is an achievement. It is not, on this corpus, and the only way to
know that is to put the constant on the same table.
"""

import argparse
import json
import math
import pathlib
import statistics
from collections import Counter, defaultdict

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


def clustered_ci(row_scores, z: float = 1.96):
    """95% interval for a mean of per-trial scores.

    Each observation is one trial's fraction of correct fields (0, .25, .5,
    .75, 1). Clustering this way costs precision versus a Wilson interval on
    every field independently - which is the point. The four fields in a trial
    are read out of one criteria paragraph by one model call; when the call
    goes wrong they go wrong together, so they are nowhere near 4N independent
    observations.
    """
    n = len(row_scores)
    if n == 0:
        return [0.0, 0.0]
    mean = statistics.fmean(row_scores)
    if n == 1:
        return [round(mean, 4), round(mean, 4)]
    se = statistics.stdev(row_scores) / math.sqrt(n)
    return [round(max(0.0, mean - z * se), 4), round(min(1.0, mean + z * se), 4)]


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


def score_group(model, prompt, group):
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
            # When it answered at all, was it right?
            "accuracy": round(hits / len(parsed), 4) if parsed else 0.0,
            "ci95": wilson(hits, len(parsed)),
            "n": len(parsed),
            # Out of every trial, parsed or not. An unparsed row is wrong.
            "accuracy_all": round(hits / n, 4) if n else 0.0,
            "ci95_all": wilson(hits, n),
            "n_all": n,
        }

    # One observation per trial: the fraction of its fields recovered.
    # An unparsed row contributes a hard zero rather than vanishing.
    row_scores = []
    for r in group:
        if r["pred"] is None:
            row_scores.append(0.0)
            continue
        correct = sum(1 for f in FIELDS
                      if field_match(f, r["gold"][f], r["pred"][f]))
        row_scores.append(correct / len(FIELDS))

    latencies = [r["latency_s"] for r in group if not r["transport_error"]]

    return {
        "model": model,
        "prompt": prompt,
        "is_baseline": model.startswith("baseline:"),
        "n_examples": n,
        "json_validity": round(len(parsed) / n, 4) if n else 0.0,
        "json_validity_ci95": wilson(len(parsed), n),
        "violations": dict(violations),
        "per_field": per_field,
        # Parsed-only. Useful for debugging a prompt, misleading as a rank.
        "mean_field_accuracy": round(
            sum(v["accuracy"] for v in per_field.values()) / len(FIELDS), 4),
        # Every trial counts. This is what the leaderboard sorts on.
        "end_to_end_accuracy": round(statistics.fmean(row_scores), 4) if row_scores else 0.0,
        "end_to_end_ci95": clustered_ci(row_scores),
        "hallucination_rate": round(hallucinated / silent_total, 4) if silent_total else None,
        "hallucination_n": silent_total,   # denominator, and it is small
        "miss_rate": round(missed / stated_total, 4) if stated_total else None,
        "miss_n": stated_total,
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "completion_tokens_total": sum(r["completion_tokens"] for r in group),
        "transport_errors": sum(1 for r in group if r["transport_error"]),
        "usd_per_1k_docs": 0.0,  # local models; set by the hosted adapter
    }


def baselines(rows):
    """Score two systems that never read the criteria text.

    If a language model cannot beat a constant, the honest headline is that it
    cannot beat a constant, and the leaderboard should say so on the same
    screen rather than in a paragraph underneath.
    """
    corpus = {}
    for r in rows:
        corpus.setdefault(r["nct_id"], r["gold"])
    gold_list = list(corpus.values())
    if not gold_list:
        return []

    majority = {}
    for field in FIELDS:
        counts = Counter(json.dumps(g[field]) for g in gold_list)
        majority[field] = json.loads(counts.most_common(1)[0][0])
    always_null = {field: None for field in FIELDS}

    out = []
    for name, constant in (("baseline:majority", majority),
                           ("baseline:all-null", always_null)):
        group = [{
            "model": name, "prompt": "constant", "nct_id": nct, "gold": gold,
            "pred": dict(constant), "violation": None, "transport_error": None,
            "latency_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
        } for nct, gold in corpus.items()]
        row = score_group(name, "constant", group)
        row["constant"] = constant
        out.append(row)
    return out


def score(raw_path=RAW, out_path=OUT):
    rows = [json.loads(l) for l in
            raw_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["prompt"])].append(row)

    results = [score_group(model, prompt, group)
               for (model, prompt), group in sorted(groups.items())]
    results.extend(baselines(rows))

    results.sort(key=lambda r: (-r["end_to_end_accuracy"], r["latency_p50_s"] or 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"scored {len(rows)} rows across {len(results)} model/prompt pairs")

    best_baseline = max((r["end_to_end_accuracy"] for r in results
                         if r.get("is_baseline")), default=0.0)
    beaten = [r for r in results
              if not r.get("is_baseline")
              and r["end_to_end_accuracy"] <= best_baseline]
    if beaten and len(beaten) == sum(1 for r in results if not r.get("is_baseline")):
        print(f"\nNOTE: no model/prompt pair beats the majority-class constant "
              f"({best_baseline:.3f}). Report that as the result.")

    dead = [r for r in results if r["json_validity"] == 0.0]
    if dead:
        print("\n*** WARNING ***")
        for r in dead:
            reason = ("every call failed before reaching the model"
                      if r["transport_errors"] == r["n_examples"]
                      else "no output parsed")
            print(f"  {r['model']} / {r['prompt']}: {reason}. "
                  f"{r['transport_errors']}/{r['n_examples']} transport errors.")
        print("These rows are not a result. Fix the cause and re-run; do not "
              "publish this page.\n")

    # A pair whose two accuracies disagree sharply is the interesting case:
    # it parsed rarely and did well on what survived. Say so out loud.
    print(f"\n  {'model':<26} {'prompt':<12} {'e2e':>6} {'parsed':>7} "
          f"{'json':>6} {'halluc':>8}")
    for r in results:
        halluc = ("    n/a" if r["hallucination_rate"] is None
                  else f"{r['hallucination_rate']:.3f}")
        flag = ""
        if r["mean_field_accuracy"] - r["end_to_end_accuracy"] > 0.15:
            flag = "  <- parsed-only figure is on a small, self-selected subset"
        elif r.get("is_baseline"):
            flag = "  <- reads nothing"
        print(f"  {r['model']:<26} {r['prompt']:<12} "
              f"{r['end_to_end_accuracy']:.3f} {r['mean_field_accuracy']:>7.3f} "
              f"{r['json_validity']:>6.3f} {halluc:>8}{flag}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=pathlib.Path, default=RAW)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    score(args.raw, args.out)


if __name__ == "__main__":
    main()
