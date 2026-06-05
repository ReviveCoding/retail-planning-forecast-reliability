import pandas as pd
from src.utils import project_path


def test_plan_has_distinct_plan_and_forecast():
    df = pd.read_csv(project_path('outputs/plans/synthetic_plan.csv'))
    assert {'plan_units','units_forecast','plan_revenue','actual_revenue'}.issubset(df.columns)
    assert (df['plan_units'] - df['units_forecast']).abs().mean() > 0
