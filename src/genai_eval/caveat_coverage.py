from __future__ import annotations


def caveat_coverage(narrative: str, caveats: list[str]) -> float:
    if not caveats:
        return 1.0
    return sum(1 for c in caveats if c.lower() in narrative.lower()) / len(caveats)
