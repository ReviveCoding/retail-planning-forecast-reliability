from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_interventions(plan_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    scenarios = [
        ('base', 1.00, 1.00),
        ('safety_stock_plus_10pct', 1.10, 1.00),
        ('capacity_plus_15pct', 1.00, 1.15),
        ('balanced_buffer_plus_5pct', 1.05, 1.05),
    ]
    for name, inv_mult, cap_mult in scenarios:
        tmp = plan_df.copy()
        tmp['scenario'] = name
        tmp['scenario_inventory'] = tmp['plan_inventory'] * inv_mult
        tmp['scenario_capacity'] = tmp['plan_capacity'] * cap_mult
        tmp['lost_sales_proxy_units'] = (tmp['actual_units'] - tmp['scenario_inventory']).clip(lower=0)
        tmp['overstock_proxy_units'] = (tmp['scenario_inventory'] - tmp['actual_units']).clip(lower=0)
        tmp['capacity_pressure_proxy'] = tmp['actual_units'] / tmp['scenario_capacity'].replace(0, 1)
        tmp['service_level_proxy'] = 1 - tmp['lost_sales_proxy_units'] / tmp['actual_units'].clip(lower=1)
        agg = tmp.groupby('scenario', as_index=False).agg(
            service_level_proxy=('service_level_proxy','mean'),
            lost_sales_proxy_units=('lost_sales_proxy_units','sum'),
            overstock_proxy_units=('overstock_proxy_units','sum'),
            capacity_pressure_proxy=('capacity_pressure_proxy','mean'),
            actual_revenue=('actual_revenue','sum'),
            plan_revenue=('plan_revenue','sum')
        )
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def build_intervention_review_queue(plan_bridge: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Create item-store review candidates from variance bridge outputs.

    This keeps the what-if layer operationally useful: the aggregate scenarios explain
    overall tradeoffs, while this queue says which SKU-store rows need review first.
    """
    df = plan_bridge.copy()
    component_cols = ['unit_effect', 'price_effect', 'interaction_effect', 'inventory_proxy_effect', 'capacity_proxy_effect', 'residual_effect']
    existing = [c for c in component_cols if c in df.columns]
    comp_abs = df[existing].abs()
    df['top_gap_component'] = comp_abs.idxmax(axis=1)
    df['gap_severity'] = df['total_revenue_gap'].abs()
    df['service_level_proxy'] = 1 - (df.get('lost_sales_proxy_units', 0) / df['actual_units'].clip(lower=1))
    df['service_level_proxy'] = df['service_level_proxy'].clip(0, 1)
    df['capacity_pressure_proxy'] = df['actual_units'] / df['plan_capacity'].replace(0, np.nan)
    df['capacity_pressure_proxy'] = df['capacity_pressure_proxy'].replace([np.inf, -np.inf], np.nan).fillna(0)

    def rec(row):
        comp = row['top_gap_component']
        if comp == 'inventory_proxy_effect' or row.get('lost_sales_proxy_units', 0) > 0:
            return 'review_safety_stock_or_inventory_target'
        if comp == 'capacity_proxy_effect' or row.get('capacity_pressure_proxy', 0) > 1:
            return 'review_capacity_buffer'
        if comp == 'price_effect':
            return 'review_price_plan_assumption'
        if comp == 'unit_effect':
            return 'review_demand_forecast_and_unit_plan'
        return 'review_residual_driver_diagnosis'

    df['recommended_intervention'] = df.apply(rec, axis=1)
    df['expected_service_level_gain'] = (1 - df['service_level_proxy']).clip(lower=0)
    df['expected_overstock_cost_change'] = df.get('overstock_proxy_units', 0) * df.get('plan_price', 1)
    sev_rank = df['gap_severity'].rank(pct=True)
    sl_rank = df['expected_service_level_gain'].rank(pct=True)
    cap_rank = df['capacity_pressure_proxy'].rank(pct=True)
    df['review_priority'] = (0.55 * sev_rank + 0.30 * sl_rank + 0.15 * cap_rank).clip(0, 1)
    keep = [
        'item_id','store_id','category_id','region_id','week_id','target_week_id','forecast_horizon',
        'total_revenue_gap','top_gap_component','recommended_intervention','expected_service_level_gain',
        'expected_overstock_cost_change','capacity_pressure_proxy','review_priority'
    ]
    keep = [c for c in keep if c in df.columns]
    return df.sort_values('review_priority', ascending=False)[keep].head(top_n).reset_index(drop=True)
