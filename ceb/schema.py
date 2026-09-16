"""The extraction target.

Four fields, chosen for one reason: ClinicalTrials.gov stores all four as
structured data *separately* from the free-text eligibility criteria. That
gives ground truth for free, with no hand labelling, and it is ground truth
nobody can accuse of being written to flatter the model.

Every field is optional. That is the point. When the criteria prose does not
state an age limit, the correct answer is null, and a model that confidently
returns 18 has hallucinated. Measuring that is the main thing this benchmark
is for.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

SEX = Literal["ALL", "FEMALE", "MALE"]


class Extraction(BaseModel):
    min_age_years: Optional[float] = Field(
        None, description="Minimum age in years, or null if the text does not say."
    )
    max_age_years: Optional[float] = Field(
        None, description="Maximum age in years, or null if the text does not say."
    )
    sex: Optional[SEX] = Field(
        None, description="ALL, FEMALE, MALE, or null if the text does not say."
    )
    accepts_healthy_volunteers: Optional[bool] = Field(
        None, description="True, false, or null if the text does not say."
    )

    model_config = {"extra": "forbid"}


FIELDS = list(Extraction.model_fields)

JSON_SCHEMA = Extraction.model_json_schema()


def parse(raw: str):
    """Parse model output into an Extraction.

    Returns (extraction, error). Exactly one is None. The error string is the
    violation category, which is reported per model: models fail in
    characteristically different ways and lumping them together hides that.
    """
    import json

    text = (raw or "").strip()
    if not text:
        return None, "empty_output"

    # Models wrap JSON in prose or fences more often than anyone admits.
    if "```" in text:
        chunks = text.split("```")
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk.startswith("json"):
                chunk = chunk[4:].strip()
            if chunk.startswith("{"):
                text = chunk
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, "no_json_found"
    candidate = text[start:end + 1]

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if not isinstance(data, dict):
        return None, "not_an_object"

    try:
        return Extraction(**data), None
    except ValidationError as exc:
        first = exc.errors()[0]
        kind = first.get("type", "unknown")
        if kind == "extra_forbidden":
            return None, "extra_field"
        return None, f"schema_violation:{kind}"
