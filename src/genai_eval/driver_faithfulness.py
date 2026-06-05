from __future__ import annotations


def driver_faithfulness(narrative: str, top_driver: str) -> float:
    return 1.0 if str(top_driver).lower() in narrative.lower() else 0.0
