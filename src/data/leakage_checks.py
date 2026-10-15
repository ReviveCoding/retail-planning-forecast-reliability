from __future__ import annotations

import numpy as np
import pandas as pd


def check_time_split(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Require finite, integral week identifiers and nonempty temporal splits."""
    train_weeks = pd.to_numeric(train["week_id"], errors="coerce").to_numpy(dtype=float)
    test_weeks = pd.to_numeric(test["week_id"], errors="coerce").to_numpy(dtype=float)
    if (
        not len(train_weeks)
        or not len(test_weeks)
        or not np.isfinite(train_weeks).all()
        or not np.isfinite(test_weeks).all()
        or not np.equal(train_weeks, np.floor(train_weeks)).all()
        or not np.equal(test_weeks, np.floor(test_weeks)).all()
    ):
        return {"valid": False, "max_train_week": None, "min_test_week": None}
    max_train = int(train_weeks.max())
    min_test = int(test_weeks.min())
    return {"valid": max_train < min_test, "max_train_week": max_train, "min_test_week": min_test}


def assert_no_future_columns(feature_cols: list[str]) -> bool:
    """Catch case-insensitive target and future feature leakage indicators."""
    return not any(
        "target" in name.casefold() or "future" in name.casefold()
        for name in feature_cols
    )
