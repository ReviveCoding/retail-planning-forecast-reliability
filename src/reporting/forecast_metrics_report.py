from __future__ import annotations
import pandas as pd
from src.utils import write_md


def write_forecast_metrics_report(metrics: pd.DataFrame, path='reports/forecast_metrics_report.md'):
    best = metrics.sort_values('wape').head(1).to_dict('records')[0] if len(metrics) else {}
    text = "# Forecast Metrics Report\n\n"
    if best:
        text += f"Best WAPE model: `{best.get('model')}` at horizon `{best.get('forecast_horizon')}` with WAPE `{best.get('wape'):.4f}`.\n\n"
    text += metrics.to_markdown(index=False) if len(metrics) else "No metrics available."
    write_md(text, path)
