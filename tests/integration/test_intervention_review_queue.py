import pandas as pd
from src.utils import project_path


def test_intervention_review_queue_exists_and_has_recommendations():
    path = project_path('outputs/plans/intervention_review_queue.csv')
    assert path.exists()
    df = pd.read_csv(path)
    assert len(df) > 0
    assert {'recommended_intervention','review_priority','top_gap_component'}.issubset(df.columns)
    assert ((df['review_priority'] >= 0) & (df['review_priority'] <= 1)).all()
