from __future__ import annotations


def generate_template_narrative(variance_summary: dict, top_driver: dict | None = None) -> str:
    top_driver = top_driver or {'driver': 'not_available', 'driver_confidence': 0.0}
    text = (
        f"Executive planning summary: total Plan-vs-Actual revenue gap is {variance_summary.get('total_revenue_gap', 0):.2f}. "
        f"The largest variance component is {variance_summary.get('largest_component', 'not_available')} "
        f"with value {variance_summary.get('largest_component_value', 0):.2f}. "
        f"The top validated demand driver is {top_driver.get('driver', 'not_available')} "
        f"with confidence {top_driver.get('driver_confidence', 0):.2f}. "
        "This summary is based on public or sample retail data and synthetic/proxy planning assumptions. "
        "It is produced by a local/offline analytical framework and should be reviewed by a human before operational use."
    )
    return text


def generate_corrupted_narratives(clean_narrative: str, variance_summary: dict, top_driver: dict | None = None) -> list[dict]:
    """Create controlled negative examples for narrative-evaluation tests.

    These are not LLM outputs. They are deterministic stress cases that verify
    whether the evaluation layer catches numeric hallucination, driver swapping,
    unsupported claims, and missing caveats.
    """
    top_driver = top_driver or {'driver': 'not_available', 'driver_confidence': 0.0}
    wrong_gap = variance_summary.get('total_revenue_gap', 0) * 1.37 + 123.45
    wrong_driver = 'unsupported_macro_driver' if top_driver.get('driver') != 'unsupported_macro_driver' else 'promotion_flag'
    return [
        {'case_id': 'clean_template', 'narrative': clean_narrative, 'expected_issue': 'none'},
        {'case_id': 'numeric_mutation', 'narrative': clean_narrative.replace(f"{variance_summary.get('total_revenue_gap', 0):.2f}", f"{wrong_gap:.2f}"), 'expected_issue': 'numeric_inconsistency'},
        {'case_id': 'driver_swap', 'narrative': clean_narrative.replace(str(top_driver.get('driver', 'not_available')), wrong_driver), 'expected_issue': 'driver_unfaithful'},
        {'case_id': 'caveat_removed', 'narrative': clean_narrative.replace('This summary is based on public or sample retail data and synthetic/proxy planning assumptions. ', ''), 'expected_issue': 'missing_caveat'},
        {'case_id': 'unsupported_claim_injected', 'narrative': clean_narrative + ' This is based on proprietary Amazon data and proves causal effects.', 'expected_issue': 'unsupported_claim'},
    ]
