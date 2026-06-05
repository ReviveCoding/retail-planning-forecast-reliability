from __future__ import annotations
import pandas as pd


def check_time_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    max_train = int(train['week_id'].max())
    min_test = int(test['week_id'].min())
    return {'valid': max_train < min_test, 'max_train_week': max_train, 'min_test_week': min_test}


def assert_no_future_columns(feature_cols: list[str]) -> bool:
    forbidden = [c for c in feature_cols if 'target' in c or 'future' in c]
    return len(forbidden) == 0
