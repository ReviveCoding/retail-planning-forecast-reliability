from __future__ import annotations
import pandas as pd


def add_lag_rolling_features(df: pd.DataFrame, lags=(1,2,4,8), windows=(4,8)) -> pd.DataFrame:
    out = df.sort_values(['item_id','store_id','week_id']).copy()
    g = out.groupby(['item_id','store_id'])['units_actual']
    for lag in lags:
        out[f'lag_{lag}'] = g.shift(lag)
    for win in windows:
        out[f'rolling_mean_{win}'] = out.groupby(['item_id','store_id'])['units_actual'].transform(lambda s: s.shift(1).rolling(win).mean())
        out[f'rolling_std_{win}'] = out.groupby(['item_id','store_id'])['units_actual'].transform(lambda s: s.shift(1).rolling(win).std())
    out['price_lag_1'] = out.groupby(['item_id','store_id'])['price_actual'].shift(1)
    return out


def make_supervised(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = add_lag_rolling_features(df)
    g_units = out.groupby(['item_id','store_id'])['units_actual']
    g_price = out.groupby(['item_id','store_id'])['price_actual']
    out['target_units'] = g_units.shift(-horizon)
    out['target_price'] = g_price.shift(-horizon)
    out['target_week_id'] = out['week_id'] + horizon
    out['forecast_horizon'] = horizon
    return out.dropna(subset=['target_units','target_price','lag_1','rolling_mean_4']).reset_index(drop=True)
