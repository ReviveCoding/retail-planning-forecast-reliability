import pandas as pd
from src.utils import project_path


def test_confidence_calibration_exists():
    df = pd.read_csv(project_path('outputs/confidence/confidence_calibration.csv'))
    assert 'empirical_pass_rate' in df.columns
