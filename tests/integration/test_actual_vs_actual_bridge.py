import pandas as pd
from src.utils import project_path


def test_actual_vs_actual_uses_distinct_prior_baseline():
    df = pd.read_csv(project_path('outputs/variance_bridge/actual_vs_actual_bridge.csv'))
    assert (df['actual_vs_prior_gap'].abs().sum() > 0)
    assert ((df['prior_units'] - df['actual_units']).abs().mean() > 0)
