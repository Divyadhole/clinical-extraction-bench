# clinical-extraction-bench

Can a small model running on a laptop recover structured eligibility
constraints from free-text clinical trial criteria &mdash; and how often does it
invent an answer the text never gave?

**[Leaderboard](https://divyadhole.github.io/clinical-extraction-bench/)**

---

## Why the ground truth is free

ClinicalTrials.gov stores `minimumAge`, `maximumAge`, `sex` and
`healthyVolunteers` as structured registry fields, **separately** from the
free-text `eligibilityCriteria`. The benchmark hides the structured fields and
gives the model only the prose.

That means no hand labelling for the core metrics, and labels that were not
written by the person grading the models.

It also produces the measurement this project exists for. Many trials never
state an age limit in their prose at all. For those, the correct answer is
`null`, and a model that returns `18` has hallucinated. Plain accuracy hides
that. **Hallucination rate** does not.

## Run it

```bash
pip install -r requirements.txt
ollama pull qwen2.5:3b && ollama pull llama3.2:3b

make fetch    # 60 trials from the API
make bench    # every model x every prompt, resumable
make score
make report   # writes docs/index.html
```

No API key, no account, no cost. Everything runs locally through Ollama.
The hosted-model adapter is written and switched off in `ceb/runners.py`;
enabling it is a config line when a paid comparison is worth the few dollars.

## What is measured

| Metric | Why it is here |
|---|---|
| Accuracy, per field | The four fields fail at very different rates. One averaged number hides which. |
| `json_validity` | How often output parsed *and* satisfied the schema. A model with better accuracy on parseable rows and worse validity is often the worse model. |
| `hallucination_rate` | Of the cases where the prose states nothing, how often the model supplied a value anyway. The headline number. |
| `miss_rate` | The reverse: a value was stated and the model returned `null`. |
| Violation categories | `no_json_found`, `invalid_json`, `extra_field`, `schema_violation` &mdash; models fail in characteristically different ways. |
| Latency P50 / P95 | Never a mean. The tail is what breaks a service. |

Intervals are Wilson 95%. At n=60 they are wide, the leaderboard says so, and
no ranking inside an overlapping interval is a real result.

## Three prompts, not one

`zero_shot`, `few_shot` and `decomposed` (`ceb/prompts.py`). Reporting one
accuracy figure for one prompt cannot distinguish the model's contribution
from the prompt's. Decoding is `temperature=0` with a fixed seed: a benchmark
whose numbers move between runs is not a benchmark.

## How good is the free ground truth?

Registry metadata is not always supported by the criteria prose. A benchmark
built on found labels has to say how wrong its labels are, or every number
above it is uninterpretable.

```bash
make audit    # ~20 minutes, 40 trials, one field at a time
```

It prints the share of registry values the prose actually supports. That
figure is the benchmark's ceiling and belongs in this README next to the
results.

## What this is not

- **Not a medical device, and not clinical advice.** Recovering a stated age
  range is not screening a patient for a trial.
- **Not a model ranking of general capability.** It measures one narrow
  extraction task on one corpus.
- **No RAG, no vector database, no agents.** Every trial's criteria fit in one
  prompt. There is no retrieval problem here, and adding one would only pad
  the tooling list.

## Tests

```bash
pytest
```

Nine tests, no network and no Ollama required: a fake runner drives the whole
pipeline end to end, including a deliberately sloppy model that emits fenced
JSON, refusals and schema violations, so the scoring path is exercised against
the failures it exists to count.

## Layout

```
ceb/schema.py    the four-field target, and output parsing with violation categories
ceb/age.py       "6 Months" and "18 Years" into comparable floats
ceb/prompts.py   the three prompts
ceb/runners.py   Ollama adapter, hosted adapter, fake runner for tests
ceb/fetch.py     corpus from the ClinicalTrials.gov v2 API
ceb/bench.py     model x prompt x example, resumable
ceb/score.py     metrics with Wilson intervals
ceb/report.py    static leaderboard
ceb/audit.py     measures the quality of the free ground truth
```
