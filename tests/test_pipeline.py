"""End to end over a fake model. No network, no Ollama, no API key."""

import json
import pathlib

from ceb import bench, report, score
from ceb.runners import EchoRunner


CORPUS = [
    # gold says 18+, no upper bound, ALL, healthy volunteers unknown
    {"nct_id": "NCT1", "title": "A", "criteria": "Inclusion: adults 18 and older.",
     "gold": {"min_age_years": 18.0, "max_age_years": None,
              "sex": "ALL", "accepts_healthy_volunteers": None}},
    # gold is silent on everything: the abstention case
    {"nct_id": "NCT2", "title": "B", "criteria": "Inclusion: documented anaemia.",
     "gold": {"min_age_years": None, "max_age_years": None,
              "sex": None, "accepts_healthy_volunteers": None}},
]


def _write(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(e) for e in CORPUS), encoding="utf-8")
    return corpus


def test_full_run_scores_and_renders(tmp_path, monkeypatch):
    # A model that always answers 18 / ALL: correct on NCT1, hallucinating on NCT2.
    scripted = ['{"min_age_years": 18, "max_age_years": null, "sex": "ALL", '
                '"accepts_healthy_volunteers": null}']
    monkeypatch.setattr(bench.runners, "build",
                        lambda spec: EchoRunner(spec, scripted))

    raw = tmp_path / "raw.jsonl"
    bench.run(["echo"], ["zero_shot"], _write(tmp_path), raw_path=raw)

    rows = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    assert len(rows) == 2

    out = tmp_path / "scores.json"
    results = score.score(raw_path=raw, out_path=out)
    assert len(results) == 1
    r = results[0]

    assert r["json_validity"] == 1.0
    # min_age: right on NCT1, wrong on NCT2 -> 0.5
    assert r["per_field"]["min_age_years"]["accuracy"] == 0.5
    # max_age and healthy_volunteers are null in gold and in pred -> perfect
    assert r["per_field"]["max_age_years"]["accuracy"] == 1.0
    # Six gold nulls across the two examples (NCT1: max, healthy; NCT2: all four).
    # The model filled two of them - min_age and sex on NCT2, where the criteria
    # say nothing about either. That is precisely the failure being measured.
    assert r["hallucination_rate"] == round(2 / 6, 4)
    assert r["miss_rate"] == 0.0

    page = report.render(results)
    assert "clinical-extraction-bench" in page
    assert "Hallucination rate" in page


def test_bench_is_resumable(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.runners, "build",
                        lambda spec: EchoRunner(spec, ['{"sex": "ALL"}']))
    raw = tmp_path / "raw.jsonl"
    corpus = _write(tmp_path)
    bench.run(["echo"], ["zero_shot"], corpus, raw_path=raw)
    first = len(raw.read_text().splitlines())
    bench.run(["echo"], ["zero_shot"], corpus, raw_path=raw)
    assert len(raw.read_text().splitlines()) == first, "re-running must add nothing"


def test_unparseable_output_is_recorded_not_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(bench.runners, "build",
                        lambda spec: EchoRunner(spec, ["I'm sorry, I can't."]))
    raw = tmp_path / "raw.jsonl"
    bench.run(["echo"], ["zero_shot"], _write(tmp_path), raw_path=raw)
    results = score.score(raw_path=raw, out_path=tmp_path / "s.json")
    assert results[0]["json_validity"] == 0.0
    assert results[0]["violations"] == {"no_json_found": 2}
