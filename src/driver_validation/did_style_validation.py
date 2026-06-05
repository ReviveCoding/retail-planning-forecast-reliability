from __future__ import annotations
import numpy as np
import pandas as pd


def _safe_slope(y) -> float:
    vals = pd.Series(y).dropna().astype(float).values
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    return float(np.polyfit(x, vals, 1)[0])


def did_style_effect(df: pd.DataFrame, treatment_col: str, metric_col='units_actual') -> pd.DataFrame:
    d = df.copy().sort_values(['item_id','store_id','week_id'])
    if treatment_col not in d.columns:
        return pd.DataFrame([{'driver': treatment_col, 'did_effect': 0.0, 'treated_n': 0, 'control_n': len(d), 'pretrend_abs_diff': 0.0}])
    d['post'] = d[treatment_col].astype(int)
    d['lag_metric'] = d.groupby(['item_id','store_id'])[metric_col].shift(1)
    treated = d[d['post'] == 1]
    control = d[d['post'] == 0]
    if treated.empty or control.empty:
        return pd.DataFrame([{'driver': treatment_col, 'did_effect': 0.0, 'treated_n': len(treated), 'control_n': len(control), 'pretrend_abs_diff': 1.0}])
    treated_effect = treated[metric_col].mean() - treated['lag_metric'].mean()
    control_effect = control[metric_col].mean() - control['lag_metric'].mean()

    # Simple pre-trend diagnostic: compare average prior-slope between units that ever receive treatment and units that do not.
    ever = d.groupby(['item_id','store_id'])[treatment_col].max().rename('ever_treated')
    d = d.merge(ever, on=['item_id','store_id'], how='left')
    slopes = d.groupby(['item_id','store_id','ever_treated'])[metric_col].apply(_safe_slope).reset_index(name='pre_slope')
    treated_slope = slopes.loc[slopes['ever_treated'].eq(1), 'pre_slope'].mean()
    control_slope = slopes.loc[slopes['ever_treated'].eq(0), 'pre_slope'].mean()
    pretrend_abs_diff = float(abs((treated_slope if pd.notna(treated_slope) else 0) - (control_slope if pd.notna(control_slope) else 0)))
    return pd.DataFrame([{'driver': treatment_col, 'did_effect': float(treated_effect - control_effect), 'treated_n': len(treated), 'control_n': len(control), 'pretrend_abs_diff': pretrend_abs_diff}])
