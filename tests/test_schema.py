from ceb.schema import parse
from ceb.age import to_years, close


def test_clean_json_parses():
    e, err = parse('{"min_age_years": 18, "max_age_years": null, '
                   '"sex": "ALL", "accepts_healthy_volunteers": true}')
    assert err is None
    assert e.min_age_years == 18 and e.max_age_years is None
    assert e.sex == "ALL" and e.accepts_healthy_volunteers is True


def test_fenced_json_is_recovered():
    e, err = parse('Sure!\n```json\n{"min_age_years": 21}\n```\nHope that helps.')
    assert err is None and e.min_age_years == 21


def test_prose_wrapped_json_is_recovered():
    e, err = parse('The answer is {"sex": "FEMALE"} based on the criteria.')
    assert err is None and e.sex == "FEMALE"


def test_violations_are_categorised():
    assert parse("")[1] == "empty_output"
    assert parse("I cannot help with that.")[1] == "no_json_found"
    assert parse("{not json,}")[1] == "invalid_json"
    assert parse('{"sex": "BOTH"}')[1].startswith("schema_violation")
    assert parse('{"min_age_years": 5, "notes": "hi"}')[1] == "extra_field"


def test_age_units_normalise_to_years():
    assert to_years("18 Years") == 18
    assert abs(to_years("6 Months") - 0.5) < 1e-6
    assert to_years("N/A") is None
    assert to_years(None) is None
    assert to_years(30) == 30


def test_null_matching_is_symmetric():
    assert close(None, None)
    assert not close(18.0, None)
    assert not close(None, 18.0)
    assert close(18.0, 18.0)
