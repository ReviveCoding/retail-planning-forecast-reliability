from __future__ import annotations
import pandas as pd
from src.utils import write_md


def write_driver_validation_report(scores: pd.DataFrame, calib: pd.DataFrame, path='reports/driver_validation_report.md'):
    text = "# Driver Validation and Confidence Scoring Report\n\n"
    text += "This report treats driver outputs as validated hypotheses, not causal proof. It combines event-study lift, DiD-style comparison, pre-trend diagnostics, bootstrap intervals, placebo stress tests, temporal stability, and business-impact weighting.\n\n"
    if len(scores):
        top = scores.iloc[0]
        text += f"Top driver: `{top['driver']}` with confidence `{top['driver_confidence']:.3f}`, placebo false-positive rate `{top['placebo_false_positive_rate']:.3f}`, and pre-trend absolute difference `{top.get('pretrend_abs_diff',0):.3f}`.\n\n"
        text += "## Driver scores\n\n" + scores.to_markdown(index=False) + "\n\n"
    text += "## Confidence calibration\n\n" + (calib.to_markdown(index=False) if len(calib) else "No calibration rows.")
    write_md(text, path)
