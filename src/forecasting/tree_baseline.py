from __future__ import annotations
import pandas as pd
from src.forecasting.simple_ridge import ridge_fit_predict

FEATURE_COLS = ['lag_1','lag_2','lag_4','lag_8','rolling_mean_4','rolling_mean_8','rolling_std_4','price_actual','price_lag_1','price_change_1','event_flag','holiday_flag','promotion_flag','week_sin','week_cos','category_id','region_id','demand_state_timestep']


def train_tree_forecast(supervised: pd.DataFrame, test_weeks=12, random_state=42) -> pd.DataFrame:
    """A lightweight tree-baseline replacement using regularized tabular regression.

    The original project used a scikit-learn tree model. This implementation keeps the
    same API and baseline role while avoiding heavy background workers, which makes the
    repo more predictable on Windows, GitHub Actions, and constrained local machines.
    """
    max_week = supervised['week_id'].max()
    split_week = max_week - test_weeks
    train = supervised[supervised['week_id'] <= split_week].copy()
    test = supervised[supervised['week_id'] > split_week].copy()
    feature_cols = [c for c in FEATURE_COLS if c in supervised.columns]
    pred = ridge_fit_predict(train, test, feature_cols, target_col='target_units', alpha=1.0)
    test['units_forecast'] = pd.Series(pred, index=test.index).clip(lower=0)
    test['model'] = 'tabular_ridge_baseline'
    test.attrs['feature_cols'] = feature_cols
    return test
