from __future__ import annotations
import pandas as pd


def compute_plan_vs_actual_bridge(plan_df: pd.DataFrame) -> pd.DataFrame:
    out = plan_df.copy()
    out['total_revenue_gap'] = out['actual_revenue'] - out['plan_revenue']
    out['unit_effect'] = (out['actual_units'] - out['plan_units']) * out['plan_price']
    out['price_effect'] = (out['actual_price'] - out['plan_price']) * out['plan_units']
    out['interaction_effect'] = (out['actual_units'] - out['plan_units']) * (out['actual_price'] - out['plan_price'])
    # proxy effects are diagnostic components, but residual preserves exact accounting identity.
    out['inventory_proxy_effect'] = -out.get('lost_sales_proxy_units', 0) * out['plan_price']
    out['capacity_proxy_effect'] = -out.get('unmet_capacity_proxy_units', 0) * out['plan_price']
    known = out['unit_effect'] + out['price_effect'] + out['interaction_effect'] + out['inventory_proxy_effect'] + out['capacity_proxy_effect']
    out['residual_effect'] = out['total_revenue_gap'] - known
    out['identity_error'] = out['total_revenue_gap'] - (known + out['residual_effect'])
    keep = ['item_id','store_id','category_id','region_id','week_id','target_week_id','forecast_horizon','model',
            'actual_units','plan_units','actual_price','plan_price','actual_revenue','plan_revenue',
            'total_revenue_gap','unit_effect','price_effect','interaction_effect','inventory_proxy_effect','capacity_proxy_effect','residual_effect','identity_error']
    return out[[c for c in keep if c in out.columns]]
