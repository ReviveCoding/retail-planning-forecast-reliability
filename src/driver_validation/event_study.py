from __future__ import annotations
import pandas as pd


def event_study_details(df: pd.DataFrame, treatment_col: str, metric_col='units_actual', pre_window=2, post_window=2) -> pd.DataFrame:
    rows = []
    if treatment_col not in df.columns:
        return pd.DataFrame(columns=['item_id','store_id','driver','before_mean','after_mean','relative_lift'])
    for (item, store), g in df.sort_values('week_id').groupby(['item_id','store_id']):
        g = g.reset_index(drop=True)
        event_idx = g.index[g[treatment_col].astype(int) == 1].tolist()
        for idx in event_idx:
            before = g.loc[max(0, idx-pre_window):idx-1, metric_col].mean()
            after = g.loc[idx:min(len(g)-1, idx+post_window), metric_col].mean()
            if pd.notna(before) and pd.notna(after) and before > 0:
                rows.append({'item_id': item, 'store_id': store, 'driver': treatment_col, 'before_mean': before, 'after_mean': after, 'relative_lift': (after-before)/before})
    return pd.DataFrame(rows)


def event_study(df: pd.DataFrame, treatment_col: str, metric_col='units_actual') -> pd.DataFrame:
    raw = event_study_details(df, treatment_col, metric_col)
    if raw.empty:
        return pd.DataFrame(columns=['driver','effect_mean','effect_median','n_events'])
    agg = raw.groupby('driver', as_index=False).agg(effect_mean=('relative_lift','mean'), effect_median=('relative_lift','median'), n_events=('relative_lift','size'))
    return agg
