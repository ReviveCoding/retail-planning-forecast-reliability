from __future__ import annotations
import re

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    vals = []
    for match in _NUMBER_RE.findall(text):
        try:
            vals.append(float(match.replace(',', '')))
        except ValueError:
            continue
    return vals


def numeric_consistency(narrative: str, expected_numbers: list[float], tolerance_digits: int = 2) -> float:
    """Score whether expected source-table numbers appear in narrative within tolerance.

    This accepts common formatting changes such as commas and minor rounding, making it
    more realistic than plain string matching for LLM-generated summaries.
    """
    if not expected_numbers:
        return 1.0
    observed = _numbers(narrative)
    if not observed:
        return 0.0
    abs_tol = 10 ** (-tolerance_digits) + 1e-9
    hits = 0
    for expected in expected_numbers:
        if any(abs(obs - expected) <= abs_tol for obs in observed):
            hits += 1
    return hits / len(expected_numbers)
