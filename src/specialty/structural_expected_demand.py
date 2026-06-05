from __future__ import annotations
import pandas as pd
from src.forecasting.simple_ridge import ridge_fit_predict

STRUCTURAL_FEATURES = ['week_sin','week_cos','price_actual','event_flag','holiday_flag','promotion_flag','category_id','region_id']


class StructuralDemandModel:
    def __init__(self, train: pd.DataFrame, feature_cols: list[str], alpha: float = 1.0):
        self.train = train.copy()
        self.feature_cols = feature_cols
        self.alpha = alpha

    def predict(self, df: pd.DataFrame):
        return pd.Series(
            ridge_fit_predict(self.train, df.copy(), self.feature_cols, target_col='target_units', alpha=self.alpha),
            index=df.index,
        )


def fit_structural_expected_demand(supervised: pd.DataFrame, test_weeks=12):
    max_week = supervised['week_id'].max()
    split_week = max_week - test_weeks
    train = supervised[supervised['week_id'] <= split_week].copy()
    test = supervised[supervised['week_id'] > split_week].copy()
    cols = [c for c in STRUCTURAL_FEATURES if c in supervised.columns]
    model = StructuralDemandModel(train, cols, alpha=1.0)
    test['expected_demand_path'] = model.predict(test).clip(lower=0)
    return model, test
