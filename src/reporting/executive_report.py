from __future__ import annotations
import pandas as pd
from src.utils import write_md


def write_executive_report(forecast_metrics: pd.DataFrame, plan_bridge: pd.DataFrame, driver_scores: pd.DataFrame, intervention: pd.DataFrame, narrative: str, path='reports/executive_ltp_review.md'):
    best = forecast_metrics.sort_values('wape').head(1).to_dict('records')[0] if len(forecast_metrics) else {}
    total_gap = plan_bridge['total_revenue_gap'].sum() if len(plan_bridge) else 0
    top_driver = driver_scores.iloc[0].to_dict() if len(driver_scores) else {'driver':'not_available','driver_confidence':0}
    text = "# Executive LTP-Style Planning Review\n\n"
    text += "## 1. Forecast performance\n\n"
    text += f"Best model by WAPE: `{best.get('model','not_available')}` at horizon `{best.get('forecast_horizon','n/a')}` with WAPE `{best.get('wape',0):.4f}`.\n\n"
    text += "## 2. Plan-vs-Actual variance\n\n"
    text += f"Total Plan-vs-Actual revenue gap across evaluated rows: `{total_gap:,.2f}`.\n\n"
    text += "## 3. Top driver hypothesis\n\n"
    text += f"Top driver: `{top_driver.get('driver')}` with confidence `{top_driver.get('driver_confidence',0):.3f}`.\n\n"
    text += "## 4. Intervention scenarios\n\n"
    if len(intervention):
        text += intervention.to_markdown(index=False) + "\n\n"
    text += "## 5. Generated planning narrative\n\n"
    text += narrative + "\n\n"
    text += "## 6. Claim boundary\n\nPublic/sample retail data are used. Planning, inventory, capacity, and intervention variables are synthetic/proxy assumptions. This is a local/offline analytical framework, not an internal production system.\n"
    write_md(text, path)
