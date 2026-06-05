from __future__ import annotations
import pandas as pd


def forecast_seasonal_naive(supervised: pd.DataFrame, horizon: int) -> pd.DataFrame:
    lag_col = 'lag_4' if 'lag_4' in supervised else 'lag_1'
    pred = supervised.copy()
    pred['units_forecast'] = pred[lag_col].clip(lower=0)
    pred['model'] = 'seasonal_naive_lag4'
    return pred
