from __future__ import annotations
import pandas as pd


def forecast_moving_average(supervised: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pred = supervised.copy()
    pred['units_forecast'] = pred['rolling_mean_4'].fillna(pred['lag_1']).clip(lower=0)
    pred['model'] = 'moving_average_4w'
    return pred
