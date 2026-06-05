from src.data.leakage_checks import assert_no_future_columns
from src.forecasting.tree_baseline import FEATURE_COLS


def test_feature_cols_do_not_contain_future_or_target():
    assert assert_no_future_columns(FEATURE_COLS)
