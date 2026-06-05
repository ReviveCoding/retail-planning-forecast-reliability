from __future__ import annotations
import pandas as pd


def generate_synthetic_plan(forecasts: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = forecasts.copy()
    event_adj = 1 + (cfg.get('event_adjustment', 1.05) - 1) * out.get('event_flag', 0)
    promo_adj = 1 + (cfg.get('promotion_plan_adjustment', 1.10) - 1) * out.get('promotion_flag', 0)
    out['plan_units'] = (
        out['units_forecast'] * cfg.get('target_growth_factor', 1.03) * event_adj * promo_adj * cfg.get('strategic_override_factor', 1.0)
    ).clip(lower=0)
    out['plan_price'] = out['price_actual'].fillna(out.get('price_lag_1', out['target_price'])) * cfg.get('planned_price_adjustment', 1.0)
    out['plan_revenue'] = out['plan_units'] * out['plan_price']
    out['actual_units'] = out['target_units']
    out['actual_price'] = out['target_price']
    out['actual_revenue'] = out['actual_units'] * out['actual_price']
    out['plan_inventory'] = out['plan_units'] * cfg.get('safety_stock_factor', 1.15)

    # Preserve a comparable historical baseline for Actual-vs-Actual analysis.
    # This is intentionally not the same as current actuals. It uses available
    # lagged/rolling information at the forecast origin, which keeps the bridge
    # closer to a business review of current actuals against recent comparable actuals.
    out['prior_units_baseline'] = out.get('rolling_mean_4', out.get('lag_1', out['units_actual'])).fillna(out.get('lag_1', out['units_actual']))
    out['prior_price_baseline'] = out.get('price_lag_1', out['price_actual']).fillna(out['price_actual'])
    out['prior_revenue_baseline'] = out['prior_units_baseline'] * out['prior_price_baseline']

    # proxy capacity based on recent rolling demand or plan units if unavailable
    base_capacity = out.get('rolling_mean_8', out['plan_units']).fillna(out['plan_units'])
    out['plan_capacity'] = base_capacity * cfg.get('capacity_buffer', 1.10)
    return out
