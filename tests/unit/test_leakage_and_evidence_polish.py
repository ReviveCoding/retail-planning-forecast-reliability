"""Regression tests for temporal leakage checks and evidence-based confidence."""

import pandas as pd

from src.data.leakage_checks import check_time_split, assert_no_future_columns
from src.driver_validation.driver_confidence_score import score_driver


def test_time_split_requires_nonempty_ordered_integral_weeks():
    train = pd.DataFrame({"week_id": [1, 2]})
    test = pd.DataFrame({"week_id": [3, 4]})
    assert check_time_split(train, test) == {
        "valid": True, "max_train_week": 2, "min_test_week": 3
    }
    assert not check_time_split(test, train)["valid"]
    assert not check_time_split(train.iloc[0:0], test)["valid"]
    assert not check_time_split(train, test.iloc[0:0])["valid"]
    assert not check_time_split(pd.DataFrame({"week_id": [1, None]}), test)["valid"]
    assert not check_time_split(train, pd.DataFrame({"week_id": [3, float("inf")]}))["valid"]
    assert not check_time_split(train, pd.DataFrame({"week_id": [3.5]}))["valid"]


def test_future_features_are_rejected_case_insensitively():
    assert assert_no_future_columns(["lag_1", "price_change_1"])
    assert not assert_no_future_columns(["lag_1", "TargetUnits"])
    assert not assert_no_future_columns(["FUTURE_revenue"])


def test_driver_confidence_is_zero_without_event_evidence():
    frame = pd.DataFrame({
        "item_id": ["i"] * 8,
        "store_id": ["s"] * 8,
        "week_id": list(range(1, 9)),
        "units_actual": [10.0] * 8,
        "promotion_flag": [0] * 8,
    })
    result = score_driver(
        frame, "promotion_flag", placebo_iterations=4, bootstrap_samples=4
    )
    assert result["bootstrap_valid_observations"] == 0
    assert result["statistical_confidence"] == 0.0


def test_driver_confidence_remains_defined_with_real_event_evidence():
    frame = pd.DataFrame({
        "item_id": ["i"] * 8,
        "store_id": ["s"] * 8,
        "week_id": list(range(1, 9)),
        "units_actual": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "promotion_flag": [0, 0, 0, 1, 0, 0, 0, 0],
    })
    result = score_driver(
        frame, "promotion_flag", placebo_iterations=4, bootstrap_samples=4
    )
    assert result["bootstrap_valid_observations"] == 1
    assert 0.0 < result["statistical_confidence"] <= 1.0
