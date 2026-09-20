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
    models = [r for r in results if not r["is_baseline"]]
    assert len(models) == 1
    # Both baselines are always scored, so the leaderboard never shows a model
    # score without a constant to compare it against.
    assert {r["model"] for r in results if r["is_baseline"]} == {
        "baseline:majority", "baseline:all-null"}
    r = models[0]

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
    r = next(r for r in results if not r["is_baseline"])
    assert r["json_validity"] == 0.0
    assert r["violations"] == {"no_json_found": 2}
    # Nothing parsed, so nothing is right, and the parsed-only average must not
    # paper over it by dividing by zero rows.
    assert r["end_to_end_accuracy"] == 0.0


class DeadRunner:
    """Every call fails at the transport layer, like a stopped Ollama."""

    kind = "fake"

    def __init__(self, *a, **kw):
        self.model = "dead"

    def healthcheck(self):
        return None

    def generate(self, prompt):
        from ceb.runners import Result
        return Result("", 0.001, error="connection refused")


def test_a_dead_server_aborts_instead_of_writing_zeros(tmp_path, monkeypatch):
    """The bug this file exists to prevent: 360 calls, zero data, exit 0."""
    import pytest

    monkeypatch.setattr(bench.runners, "build", lambda spec: DeadRunner())
    raw = tmp_path / "raw.jsonl"
    big = [dict(CORPUS[0], nct_id=f"NCT{i}") for i in range(20)]
    corpus = tmp_path / "c.jsonl"
    corpus.write_text("\n".join(json.dumps(e) for e in big), encoding="utf-8")

    with pytest.raises(RuntimeError, match="in a row failed"):
        bench.run(["dead"], ["zero_shot"], corpus, raw_path=raw)

    written = [l for l in raw.read_text().splitlines() if l.strip()]
    assert len(written) == bench.ABORT_AFTER_ERRORS, \
        "should stop at the abort threshold, not run the whole corpus"


def test_healthcheck_failure_stops_before_any_call(tmp_path, monkeypatch):
    import pytest

    class Unreachable(DeadRunner):
        def healthcheck(self):
            raise RuntimeError("Ollama is not reachable at http://localhost:11434")

    monkeypatch.setattr(bench.runners, "build", lambda spec: Unreachable())
    raw = tmp_path / "raw.jsonl"
    with pytest.raises(RuntimeError, match="not reachable"):
        bench.run(["x"], ["zero_shot"], _write(tmp_path), raw_path=raw)
    assert not raw.exists() or raw.read_text().strip() == "", \
        "nothing should be written when preflight fails"


def test_ollama_healthcheck_reports_a_missing_model(monkeypatch):
    import pytest

    from ceb import runners as R

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"models": [{"name": "llama3.2:3b"}]}

    monkeypatch.setattr(R.requests, "get", lambda *a, **kw: FakeResp())
    R.OllamaRunner("llama3.2:3b").healthcheck()          # installed: fine
    with pytest.raises(RuntimeError, match="not installed"):
        R.OllamaRunner("qwen2.5:3b").healthcheck()


def test_all_null_baseline_cannot_hallucinate():
    """The all-null constant is in the table precisely because its
    hallucination rate is zero by construction. If that ever stops being true,
    the hallucination metric is measuring something other than what it claims."""
    raw = pathlib.Path(__file__).parent / "_null.jsonl"
    raw.write_text("\n".join(json.dumps({
        "model": "echo", "prompt": "zero_shot", "nct_id": f"NCT{i}",
        "gold": {"min_age_years": 18.0, "max_age_years": None,
                 "sex": "ALL", "accepts_healthy_volunteers": True},
        "pred": None, "violation": "no_json_found", "transport_error": None,
        "latency_s": 0.1, "prompt_tokens": 1, "completion_tokens": 0,
    }) for i in range(3)), encoding="utf-8")
    try:
        results = score.score(raw_path=raw, out_path=raw.with_suffix(".json"))
        null_row = next(r for r in results if r["model"] == "baseline:all-null")
        assert null_row["hallucination_rate"] == 0.0
        assert null_row["json_validity"] == 1.0
        # It still gets credit for the fields that genuinely are unstated.
        assert null_row["per_field"]["max_age_years"]["accuracy_all"] == 1.0
    finally:
        raw.unlink(missing_ok=True)
        raw.with_suffix(".json").unlink(missing_ok=True)
