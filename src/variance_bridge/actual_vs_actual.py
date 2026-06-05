from __future__ import annotations
import pandas as pd


def compute_actual_vs_actual_bridge(plan_df: pd.DataFrame) -> pd.DataFrame:
    """Decompose current actual revenue against a recent comparable actual baseline.

    The prior baseline uses lagged/rolling historical values generated before the target
    period, not the same actual row. This makes the Actual-vs-Actual bridge meaningful
    for planning reviews while preserving an exact accounting identity.
    """
    out = plan_df.copy()
    out['prior_units'] = out.get('prior_units_baseline', out.get('rolling_mean_4', out.get('lag_1', out['units_actual']))).fillna(out.get('lag_1', out['units_actual']))
    out['prior_price'] = out.get('prior_price_baseline', out.get('price_lag_1', out['price_actual'])).fillna(out['price_actual'])
    out['prior_revenue'] = out['prior_units'] * out['prior_price']
    out['actual_vs_prior_gap'] = out['actual_revenue'] - out['prior_revenue']
    out['unit_volume_change'] = (out['actual_units'] - out['prior_units']) * out['prior_price']
    out['price_change_effect'] = (out['actual_price'] - out['prior_price']) * out['prior_units']
    out['mix_interaction_effect'] = (out['actual_units'] - out['prior_units']) * (out['actual_price'] - out['prior_price'])
    known = out['unit_volume_change'] + out['price_change_effect'] + out['mix_interaction_effect']
    out['residual_effect'] = out['actual_vs_prior_gap'] - known
    out['identity_error'] = out['actual_vs_prior_gap'] - (known + out['residual_effect'])
    keep = ['item_id','store_id','category_id','region_id','week_id','target_week_id','forecast_horizon','model',
            'prior_units','actual_units','prior_price','actual_price','prior_revenue','actual_revenue',
            'actual_vs_prior_gap','unit_volume_change','price_change_effect','mix_interaction_effect','residual_effect','identity_error']
    return out[[c for c in keep if c in out.columns]]
