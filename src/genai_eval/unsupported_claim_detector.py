from __future__ import annotations

UNSUPPORTED_PATTERNS = ['Amazon internal', 'proprietary Amazon data', 'production deployment', 'proved causal', 'guaranteed']


def unsupported_claim_rate(narrative: str, patterns: list[str] | None = None) -> float:
    patterns = patterns or UNSUPPORTED_PATTERNS
    if not patterns:
        return 0.0
    return sum(1 for p in patterns if p.lower() in narrative.lower()) / len(patterns)
