from __future__ import annotations
from src.utils import write_md


def write_claim_boundary_report(path='reports/claim_boundary.md'):
    text = """# Claim Boundary

This project uses public retail data where available or generated sample M5-like/Favorita-like data.

Planning, inventory, capacity, and intervention variables are synthetic/proxy planning assumptions.

This is a local/offline analytical framework. It is not an Amazon internal system, does not use proprietary Amazon data, and is not a production deployment.

The causal/driver module provides causal/driver-analysis validation and confidence-scored hypotheses. It does not prove true causal effects.
"""
    write_md(text, path)
    return text
