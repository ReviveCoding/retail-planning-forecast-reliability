from __future__ import annotations
import numpy as np
import pandas as pd


def add_calendar_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['week_sin'] = np.sin(2*np.pi*out['week_id']/52.0)
    out['week_cos'] = np.cos(2*np.pi*out['week_id']/52.0)
    out['price_change_1'] = out['price_actual'] / out['price_lag_1'].replace(0, np.nan) - 1
    out['price_change_1'] = out['price_change_1'].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    return out
