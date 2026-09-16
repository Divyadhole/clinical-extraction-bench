"""The three prompts under test.

The comparison is the product. One prompt with one accuracy number says
nothing about whether the prompt or the model is doing the work.

Templates use a <<CRITERIA>> sentinel and str.replace rather than str.format,
because the embedded JSON schema is full of braces and format() chokes on them.
"""

import json

from .schema import JSON_SCHEMA

_SCHEMA = json.dumps(JSON_SCHEMA, indent=2)

_RULES = """Return ONLY a JSON object. No prose, no markdown fences.

Use null for any field the text does not state. Do not infer, do not guess a
typical value, and do not carry a value over from a different field. A trial
that never mentions an upper age limit has max_age_years of null."""

ZERO_SHOT = f"""Extract the eligibility constraints from the clinical trial \
criteria below.

{_RULES}

Schema:
{_SCHEMA}

Criteria:
<<CRITERIA>>

JSON:"""

FEW_SHOT = f"""Extract the eligibility constraints from clinical trial criteria.

{_RULES}

Schema:
{_SCHEMA}

Example 1
Criteria: "Inclusion Criteria: - Adults aged 18 years and older - Male or \
female - Confirmed Type 2 diabetes. Exclusion Criteria: - Pregnancy"
JSON: {{"min_age_years": 18, "max_age_years": null, "sex": "ALL", \
"accepts_healthy_volunteers": null}}

Example 2
Criteria: "Inclusion Criteria: - Post-menopausal women between 50 and 70 \
years - Healthy volunteers welcome"
JSON: {{"min_age_years": 50, "max_age_years": 70, "sex": "FEMALE", \
"accepts_healthy_volunteers": true}}

Example 3
Criteria: "Inclusion Criteria: - Documented heart failure - Able to provide \
consent. Exclusion Criteria: - Prior transplant"
JSON: {{"min_age_years": null, "max_age_years": null, "sex": null, \
"accepts_healthy_volunteers": null}}

Now the real one.
Criteria:
<<CRITERIA>>

JSON:"""

DECOMPOSED = f"""You will extract four fields from clinical trial eligibility \
criteria. Consider each one separately and independently.

1. min_age_years - does the text state a lower age bound? "18 years and older"
   is 18. If nothing is stated, it is null.
2. max_age_years - does the text state an upper age bound? "18 and older" does
   NOT imply an upper bound. If nothing is stated, it is null.
3. sex - does the text restrict by sex? "Male or female" is ALL. A trial about
   post-menopausal women is FEMALE. If nothing is stated, it is null.
4. accepts_healthy_volunteers - does the text say healthy volunteers are
   accepted or excluded? If it is silent, it is null.

{_RULES}

Schema:
{_SCHEMA}

Criteria:
<<CRITERIA>>

JSON:"""

PROMPTS = {"zero_shot": ZERO_SHOT, "few_shot": FEW_SHOT, "decomposed": DECOMPOSED}


def render(name: str, criteria: str) -> str:
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt {name!r}; have {sorted(PROMPTS)}")
    return PROMPTS[name].replace("<<CRITERIA>>", criteria.strip())
