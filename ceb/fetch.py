"""Build the corpus from the ClinicalTrials.gov v2 API.

Ground truth is free: the registry stores minimumAge, maximumAge, sex and
healthyVolunteers as structured fields, separately from the free-text
eligibility criteria. The model sees only the prose and has to recover the
structure.

The registry's own fields are not perfect, and the README says so. `audit.py`
measures how imperfect on a sample, which is the honest way to report a
benchmark built on found labels rather than pretending the ceiling is 100%.
"""

import argparse
import json
import pathlib
import time

import requests

from .age import to_years

BASE = "https://clinicaltrials.gov/api/v2/studies"
MIN_CRITERIA_CHARS = 250


def _dig(node, *keys):
    for key in keys:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
        if node is None:
            return None
    return node


def to_example(study):
    elig = _dig(study, "protocolSection", "eligibilityModule") or {}
    ident = _dig(study, "protocolSection", "identificationModule") or {}
    criteria = (elig.get("eligibilityCriteria") or "").strip()
    nct_id = ident.get("nctId")

    if not nct_id or len(criteria) < MIN_CRITERIA_CHARS:
        return None

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle"),
        "criteria": criteria,
        "gold": {
            "min_age_years": to_years(elig.get("minimumAge")),
            "max_age_years": to_years(elig.get("maximumAge")),
            "sex": elig.get("sex"),
            "accepts_healthy_volunteers": elig.get("healthyVolunteers"),
        },
    }


def fetch(n: int, out: pathlib.Path, page_size: int = 100, sleep: float = 1.5):
    session = requests.Session()
    session.headers.update({"User-Agent": "clinical-extraction-bench/0.1"})

    params = {
        "pageSize": page_size,
        "countTotal": "true",
        "filter.overallStatus": "RECRUITING",
        "sort": "LastUpdatePostDate",
    }
    examples, seen, token = [], set(), None

    while len(examples) < n:
        if token:
            params = dict(params, pageToken=token)
            params.pop("countTotal", None)
        resp = session.get(BASE, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        for study in payload.get("studies") or []:
            example = to_example(study)
            if example and example["nct_id"] not in seen:
                seen.add(example["nct_id"])
                examples.append(example)
                if len(examples) >= n:
                    break

        token = payload.get("nextPageToken")
        print(f"  collected {len(examples)}/{n}")
        if not token:
            break
        time.sleep(sleep)  # stay well under the ~50 req/min public limit

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    nulls = {
        field: sum(1 for e in examples if e["gold"][field] is None)
        for field in examples[0]["gold"]
    } if examples else {}
    print(f"\nwrote {len(examples)} examples to {out}")
    print("gold nulls per field (these are the abstention cases):", nulls)
    return examples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=60)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("data/corpus.jsonl"))
    args = ap.parse_args()
    fetch(args.n, args.out)


if __name__ == "__main__":
    main()
