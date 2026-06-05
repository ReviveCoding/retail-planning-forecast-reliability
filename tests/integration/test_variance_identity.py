import pandas as pd
from src.utils import project_path
from src.variance_bridge.bridge_identity_tests import identity_passes, max_identity_error


def test_plan_vs_actual_identity():
    df = pd.read_csv(project_path('outputs/variance_bridge/plan_vs_actual_bridge.csv'))
    assert identity_passes(df, tolerance=1e-6), max_identity_error(df)


def test_actual_vs_actual_identity():
    df = pd.read_csv(project_path('outputs/variance_bridge/actual_vs_actual_bridge.csv'))
    assert identity_passes(df, tolerance=1e-6), max_identity_error(df)
