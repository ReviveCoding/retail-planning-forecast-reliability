import pandas as pd
from src.utils import project_path


def test_asymmetric_metrics_in_bounds():
    df = pd.read_csv(project_path('outputs/specialty/asymmetric_coverage_summary.csv'))
    for c in ['coverage_recall','buffer_precision','coverage_f1']:
        assert ((df[c] >= 0) & (df[c] <= 1.000001)).all()
