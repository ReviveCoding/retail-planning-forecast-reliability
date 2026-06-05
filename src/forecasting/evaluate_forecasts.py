from __future__ import annotations
import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    denom = np.maximum(np.abs(y_true).sum(), 1e-9)
    return {
        'rmse': float(np.sqrt(np.mean(err**2))),
        'mae': float(np.mean(np.abs(err))),
        'wape': float(np.abs(err).sum() / denom),
        'bias': float(np.mean(err)),
        'under_forecast_rate': float(np.mean(y_pred < y_true)),
        'over_forecast_rate': float(np.mean(y_pred > y_true)),
    }


def summarize_model_metrics(forecast_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, horizon), g in forecast_df.groupby(['model','forecast_horizon']):
        m = regression_metrics(g['target_units'], g['units_forecast'])
        m.update({'model': model, 'forecast_horizon': horizon, 'n': len(g)})
        rows.append(m)
    return pd.DataFrame(rows).sort_values(['forecast_horizon','model'])
