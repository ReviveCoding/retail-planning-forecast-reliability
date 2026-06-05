import pandas as pd
from src.utils import project_path


def test_forecasts_have_future_target_week():
    df = pd.read_csv(project_path('outputs/forecasts/all_forecasts.csv'))
    assert (df['target_week_id'] > df['week_id']).all()
