import pandas as pd
from src.utils import project_path


def test_gated_residual_correction_uses_validation_gate_and_reports_test_wape():
    df = pd.read_csv(project_path('outputs/specialty/specialty_ablation_metrics.csv'))
    gated = df[df['model'].eq('gated_business_mechanism_residual_correction')].iloc[0]
    assert gated['gate_dataset'] == 'validation'
    assert pd.notna(gated['validation_baseline_wape_for_gate'])
    assert pd.notna(gated['validation_raw_wape_for_gate'])
