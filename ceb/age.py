"""Age normalisation.

The registry writes ages as '18 Years', '6 Months', '30 Days', 'N/A'. Models
write them as numbers, strings, or prose. Everything becomes years as a float
so that comparison is not a string match.
"""

import re
from typing import Optional

UNITS = {
    "year": 1.0, "years": 1.0, "yr": 1.0, "yrs": 1.0,
    "month": 1 / 12, "months": 1 / 12, "mo": 1 / 12,
    "week": 1 / 52.1775, "weeks": 1 / 52.1775,
    "day": 1 / 365.25, "days": 1 / 365.25,
    "hour": 1 / 8766.0, "hours": 1 / 8766.0,
    "minute": 1 / 525960.0, "minutes": 1 / 525960.0,
}


def to_years(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text or text in {"n/a", "na", "none", "null", "unknown"}:
        return None
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([a-z]*)", text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2) or "years"
    return number * UNITS.get(unit, 1.0)


def close(a: Optional[float], b: Optional[float], tol: float = 0.02) -> bool:
    """Both null counts as agreement. One null does not."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol
