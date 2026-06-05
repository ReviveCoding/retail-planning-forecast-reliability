from __future__ import annotations


def route_for_review(metrics: dict, thresholds: dict) -> bool:
    return (
        metrics.get('numeric_consistency', 0) < thresholds.get('min_numeric_consistency', 0.95)
        or metrics.get('driver_faithfulness', 0) < thresholds.get('min_driver_faithfulness', 1.0)
        or metrics.get('caveat_coverage', 0) < thresholds.get('min_caveat_coverage', 0.95)
        or metrics.get('unsupported_claim_rate', 1) > thresholds.get('max_unsupported_claim_rate', 0.05)
    )
