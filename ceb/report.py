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
.ci { color: #8a867e; font-size: 11.5px; }
.wrap { overflow-x: auto; }
footer { margin-top: 2.5rem; color: #8a867e; font-size: 12.5px; max-width: 46rem; }
@media (prefers-color-scheme: dark) {
  body { background: #16161a; color: #e7e5e0; }
  th, td { border-bottom-color: #2c2c33; }
  th, p.sub, footer, .ci { color: #9a978f; }
}
"""


def render(scores):
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n = scores[0]["n_examples"] if scores else 0

    head = "".join(f"<th>{html.escape(f.replace('_', ' '))}</th>" for f in FIELDS)
    rows = []
    for i, r in enumerate(scores):
        cells = "".join(
            f"<td class='num'>{r['per_field'][f]['accuracy']:.2f}"
            f"<br><span class='ci'>{r['per_field'][f]['ci95'][0]:.2f}"
            f"&ndash;{r['per_field'][f]['ci95'][1]:.2f}</span></td>"
            for f in FIELDS)
        halluc = ("&mdash;" if r["hallucination_rate"] is None
                  else f"{r['hallucination_rate']:.2f}")
        rows.append(
            f"<tr class='{'best' if i == 0 else ''}'>"
            f"<td>{html.escape(r['model'])}</td>"
            f"<td>{html.escape(r['prompt'])}</td>"
            f"<td class='num'>{r['mean_field_accuracy']:.3f}</td>"
            f"<td class='num'>{r['json_validity']:.2f}</td>"
            f"<td class='num'>{halluc}</td>"
            f"<td class='num'>{r['miss_rate'] if r['miss_rate'] is not None else '&mdash;'}</td>"
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

<h2>Leaderboard</h2>
<div class="wrap"><table>
<thead><tr>
<th>Model</th><th>Prompt</th><th>Mean acc</th><th>JSON valid</th>
<th>Halluc.</th><th>Miss</th>{head}<th>P50 s</th><th>P95 s</th>
</tr></thead>
<tbody>{''.join(rows) or "<tr><td colspan='12'>No results yet.</td></tr>"}</tbody>
</table></div>

<h2>Schema violations</h2>
<div class="wrap"><table>
<thead><tr><th>Model</th><th>Prompt</th><th>Failures</th></tr></thead>
<tbody>{''.join(violations) or "<tr><td colspan='3'>Every output parsed.</td></tr>"}</tbody>
</table></div>

<footer>
<b>Hallucination rate</b> is the share of cases where the criteria text does
not state a value and the model supplied one anyway. It is the number that
matters most here and the one plain accuracy hides.
<b>Miss rate</b> is the reverse: a value was stated and the model returned null.
Intervals are Wilson 95%. With {n} examples they are wide, and no ranking
inside an overlapping interval should be read as a real difference.
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
