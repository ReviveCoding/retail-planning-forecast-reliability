from __future__ import annotations
import pandas as pd
from src.utils import write_md


def _top_components(summary: pd.Series) -> str:
    comps = ['unit_effect','price_effect','interaction_effect','inventory_proxy_effect','capacity_proxy_effect','residual_effect']
    rows = [(c, float(summary.get(c,0))) for c in comps]
    rows = sorted(rows, key=lambda x: abs(x[1]), reverse=True)
    return '\n'.join([f"- `{c}`: {v:,.2f}" for c, v in rows])


def write_variance_bridge_report(plan_bridge: pd.DataFrame, actual_bridge: pd.DataFrame, path='reports/variance_bridge_report.md'):
    s = plan_bridge[['total_revenue_gap','unit_effect','price_effect','interaction_effect','inventory_proxy_effect','capacity_proxy_effect','residual_effect','identity_error']].sum(numeric_only=True)
    a = actual_bridge[['actual_vs_prior_gap','unit_volume_change','price_change_effect','mix_interaction_effect','residual_effect','identity_error']].sum(numeric_only=True)
    text = "# Variance Bridge Report\n\n"
    text += "## Plan-vs-Actual bridge\n\n"
    text += f"Total revenue gap: `{s.get('total_revenue_gap',0):,.2f}`. Max identity error should be near zero; aggregate identity error is `{s.get('identity_error',0):.8f}`.\n\n"
    text += "Top components by absolute contribution:\n" + _top_components(s) + "\n\n"
    text += "## Actual-vs-Actual bridge\n\n"
    text += "Actual-vs-Actual uses a recent comparable actual baseline built from lagged/rolling demand and lagged price, rather than comparing a row against itself.\n\n"
    text += f"Actual-vs-prior revenue gap: `{a.get('actual_vs_prior_gap',0):,.2f}`. Aggregate identity error is `{a.get('identity_error',0):.8f}`.\n"
    write_md(text, path)
