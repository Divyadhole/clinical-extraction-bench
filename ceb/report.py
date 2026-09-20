"""Render results/scores.json as a static leaderboard in docs/index.html."""

import argparse
import datetime as dt
import html
import json
import pathlib

from .schema import FIELDS

SCORES = pathlib.Path("results/scores.json")
OUT = pathlib.Path("docs/index.html")

CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
       margin: 0; padding: 2.5rem 1.25rem; background: #fbfbfa; color: #1c1b19; }
main { max-width: 68rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2.25rem 0 .6rem; }
p.sub { color: #6b6862; margin: 0 0 1.75rem; max-width: 46rem; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #e6e3dd; }
th { font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: #6b6862; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.best td { font-weight: 600; }
tr.baseline td { background: #f2efe8; font-style: italic; }
.callout { border-left: 3px solid #b4741f; background: #faf4e9; padding: .8rem 1rem;
           margin: 0 0 1.5rem; font-size: 13.5px; max-width: 46rem; }
.ci { color: #8a867e; font-size: 11.5px; }
.wrap { overflow-x: auto; }
td.soft { color: #8a867e; }
td.warn { color: #9a5b12; }
.note { font-size: 11.5px; color: #8a867e; }
footer { margin-top: 2.5rem; color: #8a867e; font-size: 12.5px; max-width: 46rem; }
@media (prefers-color-scheme: dark) {
  body { background: #16161a; color: #e7e5e0; }
  th, td { border-bottom-color: #2c2c33; }
  th, p.sub, footer, .ci, td.soft, .note { color: #9a978f; }
  tr.baseline td { background: #1f1f26; }
  .callout { background: #241f16; border-left-color: #a97527; }
  td.warn { color: #d59a4e; }
}
"""


def render(scores):
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = scores[0]["n_examples"] if scores else 0
    nfields = len(FIELDS)
    n4 = n * nfields

    base = max((r["end_to_end_accuracy"] for r in scores if r.get("is_baseline")),
               default=0.0)
    top = max((r["end_to_end_accuracy"] for r in scores if not r.get("is_baseline")),
              default=0.0)
    callout = ""
    if scores and top <= base:
        callout = (f"<p class='callout'><b>No model beats the constant.</b> "
                   f"Always answering the most common value for each field &mdash; "
                   f"reading none of the criteria text &mdash; recovers "
                   f"{base:.1%} of fields. The best model/prompt pair here "
                   f"recovers {top:.1%}. On this corpus, at this model size, "
                   f"extraction is worse than not extracting.</p>")
    head = "".join(f"<th>{html.escape(f.replace('_', ' '))}</th>" for f in FIELDS)
    rows = []
    best_model = next((r for r in scores if not r.get("is_baseline")), None)
    for r in scores:
        # Per-field cells use the all-rows denominator so that they average to
        # the end-to-end headline instead of contradicting it.
        cells = "".join(
            f"<td class='num'>{r['per_field'][f]['accuracy_all']:.2f}"
            f"<br><span class='ci'>{r['per_field'][f]['ci95_all'][0]:.2f}"
            f"&ndash;{r['per_field'][f]['ci95_all'][1]:.2f}</span></td>"
            for f in FIELDS)

        # A big gap means the parsed-only figure was measured on a small,
        # self-selected slice. Mark it rather than letting it read as a score.
        gap = r["mean_field_accuracy"] - r["end_to_end_accuracy"]
        parsed_cls = "num warn" if gap > 0.15 else "num soft"
        parsed_note = (f"<br><span class='note'>n={int(round(r['json_validity'] * r['n_examples']))}</span>"
                       if gap > 0.15 else "")

        halluc = ("&mdash;" if r["hallucination_rate"] is None
                  else f"{r['hallucination_rate']:.2f}"
                       f"<br><span class='note'>n={r['hallucination_n']}</span>")
        miss = ("&mdash;" if r["miss_rate"] is None
                else f"{r['miss_rate']:.2f}"
                     f"<br><span class='note'>n={r['miss_n']}</span>")

        rows.append(
            f"<tr class='{'baseline' if r.get('is_baseline') else ''}"
            f"{' best' if r is best_model else ''}'>"
            f"<td>{html.escape(r['model'])}</td>"
            f"<td>{html.escape(r['prompt'])}</td>"
            f"<td class='num'>{r['end_to_end_accuracy']:.3f}"
            f"<br><span class='ci'>{r['end_to_end_ci95'][0]:.2f}"
            f"&ndash;{r['end_to_end_ci95'][1]:.2f}</span></td>"
            f"<td class='{parsed_cls}'>{r['mean_field_accuracy']:.3f}{parsed_note}</td>"
            f"<td class='num'>{r['json_validity']:.2f}</td>"
            f"<td class='num'>{halluc}</td>"
            f"<td class='num'>{miss}</td>"
            f"{cells}"
            f"<td class='num'>{r['latency_p50_s']}</td>"
            f"<td class='num'>{r['latency_p95_s']}</td>"
            f"</tr>")

    violations = []
    for r in scores:
        if r["violations"]:
            detail = ", ".join(f"{k} &times;{v}" for k, v in
                               sorted(r["violations"].items(), key=lambda kv: -kv[1]))
            violations.append(
                f"<tr><td>{html.escape(r['model'])}</td>"
                f"<td>{html.escape(r['prompt'])}</td><td>{detail}</td></tr>")

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>clinical-extraction-bench</title><style>{CSS}</style></head>
<body><main>

<h1>clinical-extraction-bench</h1>
<p class="sub">Can a small local model recover structured eligibility
constraints from free-text clinical trial criteria? Ground truth is the
registry's own structured fields, which the model never sees.
{n} trials &middot; generated {generated}.</p>

{callout}

<h2>Leaderboard</h2>
<div class="wrap"><table>
<thead><tr>
<th>Model</th><th>Prompt</th><th>End-to-end acc</th>
<th>Acc. on parsed</th><th>JSON valid</th>
<th>Halluc.</th><th>Miss</th>{head}<th>P50 s</th><th>P95 s</th>
</tr></thead>
<tbody>{''.join(rows) or "<tr><td colspan='13'>No results yet.</td></tr>"}</tbody>
</table></div>

<h2>Schema violations</h2>
<div class="wrap"><table>
<thead><tr><th>Model</th><th>Prompt</th><th>Failures</th></tr></thead>
<tbody>{''.join(violations) or "<tr><td colspan='3'>Every output parsed.</td></tr>"}</tbody>
</table></div>

<footer>
<b>End-to-end accuracy</b> is the fraction of all {n} &times; {nfields} field
values recovered, counting every field of an unparsed row as wrong. It is the
number to plan around, and it is what this table is sorted by. Its interval is
clustered by trial, not by field: the four fields come out of one model call
and fail together, so treating them as {n4} independent draws would claim
roughly twice the precision the data supports.
<b>Accuracy on parsed</b> is the older, friendlier number &mdash; hits over the
rows that produced valid JSON. It answers "when it answers, is it right?",
which is useful for debugging a prompt and misleading as a ranking, because a
model is rewarded for staying silent on the documents it finds hard. Where the
two columns diverge sharply the parsed-only figure is highlighted along with
the handful of rows it was actually measured on.
<b>Hallucination rate</b> is the share of cases where the criteria text does
not state a value and the model supplied one anyway. It is the number that
matters most here and the one plain accuracy hides. <b>Miss rate</b> is the
reverse: a value was stated and the model returned null. Both print their
denominator, because those denominators are small and shrink further every
time a model fails to parse.
Intervals are Wilson 95% except end-to-end. With {n} examples they are wide,
and no ranking inside an overlapping interval should be read as a real
difference.
Ground truth comes from registry metadata, which is itself imperfect &mdash;
see the audit section of the README for how imperfect.
</footer>
</main></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", type=pathlib.Path, default=SCORES)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(scores), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
