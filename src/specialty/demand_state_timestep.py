from __future__ import annotations
import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def add_demand_state_timestep(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add a PHT-inspired, leakage-safe demand-state timestep feature.

    The feature uses only information available up to the forecast origin:
    lagged rolling demand, expanding historical mean/std, and elapsed observed history.
    It avoids using future target demand or the final length of each series.
    """
    cfg = cfg or {}
    lifecycle_cap_weeks = float(cfg.get('lifecycle_cap_weeks', 52))
    lifecycle_cap_weeks = max(lifecycle_cap_weeks, 1.0)
    out = df.sort_values(['item_id','store_id','week_id']).copy()
    g = out.groupby(['item_id','store_id'])['units_actual']
    hist_mean = g.transform(lambda s: s.shift(1).expanding(min_periods=2).mean())
    hist_std = g.transform(lambda s: s.shift(1).expanding(min_periods=2).std()).replace(0, np.nan)
    rolling4 = g.transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    demand_z = ((rolling4 - hist_mean) / hist_std).fillna(0.0)
    state_from_demand = _sigmoid(demand_z)

    # Observed-history lifecycle progress only uses elapsed history and a config cap.
    seq_idx = out.groupby(['item_id','store_id']).cumcount().astype(float)
    lifecycle_age = (seq_idx / lifecycle_cap_weeks).fillna(0.0).clip(0, 1)

    out['demand_state_signal'] = state_from_demand
    out['demand_lifecycle_progress'] = lifecycle_age
    out['demand_state_timestep'] = (0.7 * out['demand_state_signal'] + 0.3 * out['demand_lifecycle_progress']).clip(0, 1)
    return out
