from __future__ import annotations
import pandas as pd
from src.utils import write_md


def write_specialty_ablation_report(metrics: pd.DataFrame, coverage: pd.DataFrame, path='reports/specialty_ablation_report.md'):
    text = "# Specialty-Enhanced Forecasting and Risk-Adjusted Planning Report\n\n"
    text += "This module separates three ideas: Demand-State Timestep features, structural expected demand, and gated residual correction. The gated correction is intentionally conservative: if raw residual correction underperforms the demand-state tree baseline on validation WAPE, the framework falls back to the stronger baseline rather than forcing a worse specialty model.\n\n"
    text += "## Forecast ablation metrics\n\n"
    text += metrics.to_markdown(index=False) if len(metrics) else "No specialty metrics available."
    text += "\n\n## Asymmetric coverage-vs-overreach metrics\n\n"
    text += coverage.to_markdown(index=False) if len(coverage) else "No coverage metrics available."
    write_md(text, path)
