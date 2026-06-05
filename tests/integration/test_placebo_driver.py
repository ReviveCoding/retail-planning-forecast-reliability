import pandas as pd
from src.utils import project_path


def test_driver_scores_have_confidence_bounds_and_robustness_metadata():
    df = pd.read_csv(project_path('outputs/driver_validation/driver_confidence_scores.csv'))
    assert len(df) >= 1
    assert ((df['driver_confidence'] >= 0) & (df['driver_confidence'] <= 1)).all()
    assert ((df['placebo_false_positive_rate'] >= 0) & (df['placebo_false_positive_rate'] <= 1)).all()
    assert (df['placebo_iterations'] >= 100).all()
    assert (df['bootstrap_samples'] >= 100).all()
    assert 'pretrend_abs_diff' in df.columns
