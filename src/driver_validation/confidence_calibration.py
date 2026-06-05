from __future__ import annotations
import pandas as pd


def confidence_calibration(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    s['confidence_bucket'] = pd.cut(s['driver_confidence'], bins=[0, .2, .4, .6, .8, 1.0], include_lowest=True)
    s['empirical_pass'] = ((s['placebo_false_positive_rate'] < 0.5) & (s['temporal_stability'] > 0.5)).astype(int)
    return s.groupby('confidence_bucket', observed=True).agg(
        n=('driver','size'),
        mean_confidence=('driver_confidence','mean'),
        empirical_pass_rate=('empirical_pass','mean')
    ).reset_index()
