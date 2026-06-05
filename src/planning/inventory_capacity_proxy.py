from __future__ import annotations
import pandas as pd


def add_inventory_capacity_impacts(plan_df: pd.DataFrame) -> pd.DataFrame:
    out = plan_df.copy()
    out['lost_sales_proxy_units'] = (out['actual_units'] - out['plan_inventory']).clip(lower=0)
    out['unmet_capacity_proxy_units'] = (out['actual_units'] - out['plan_capacity']).clip(lower=0)
    out['overstock_proxy_units'] = (out['plan_inventory'] - out['actual_units']).clip(lower=0)
    out['service_level_proxy'] = 1 - out['lost_sales_proxy_units'] / out['actual_units'].clip(lower=1)
    out['capacity_pressure_proxy'] = out['actual_units'] / out['plan_capacity'].replace(0, 1)
    return out
