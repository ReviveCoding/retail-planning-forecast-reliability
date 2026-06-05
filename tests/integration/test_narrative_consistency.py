import pandas as pd
from src.utils import project_path


def test_clean_narrative_eval_passes_core_checks():
    df = pd.read_csv(project_path('outputs/genai_eval/narrative_eval_metrics.csv'))
    clean = df[df['case_id'].eq('clean_template')].iloc[0]
    assert clean['numeric_consistency'] >= 0.95
    assert clean['driver_faithfulness'] == 1.0
    assert clean['human_review_required'] in [False, 0]


def test_corrupted_narratives_trigger_review_or_metric_drop():
    df = pd.read_csv(project_path('outputs/genai_eval/narrative_eval_metrics.csv'))
    corrupted = df[~df['case_id'].eq('clean_template')]
    assert len(corrupted) >= 3
    assert corrupted['human_review_required'].astype(bool).any()
    assert (corrupted[['numeric_consistency','driver_faithfulness','caveat_coverage']].min(axis=1) < 1.0).any() or (corrupted['unsupported_claim_rate'] > 0).any()
