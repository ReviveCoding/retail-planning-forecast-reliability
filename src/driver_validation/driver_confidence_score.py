from __future__ import annotations
import numpy as np
import pandas as pd
from .event_study import event_study, event_study_details
from .did_style_validation import did_style_effect
from .placebo_tests import placebo_test
from .bootstrap_ci import bootstrap_mean_ci


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def score_driver(
    df: pd.DataFrame,
    treatment_col: str,
    weights: dict | None = None,
    metric_col='units_actual',
    placebo_iterations: int = 200,
    bootstrap_samples: int = 200,
) -> dict:
    weights = weights or {
        'effect_size_strength': .25,
        'statistical_confidence': .20,
        'temporal_stability': .20,
        'placebo_pass_score': .20,
        'business_impact_weight': .10,
        'pretrend_quality': .05,
    }
    ev = event_study(df, treatment_col, metric_col)
    ev_details = event_study_details(df, treatment_col, metric_col)
    did = did_style_effect(df, treatment_col, metric_col)
    pl = placebo_test(df, treatment_col, metric_col, iterations=placebo_iterations)
    effect = float(ev['effect_mean'].iloc[0]) if len(ev) else 0.0
    n_events = int(ev['n_events'].iloc[0]) if len(ev) and 'n_events' in ev else 0
    effect_values = ev_details['relative_lift'].values if len(ev_details) else []
    mean, lo, hi = bootstrap_mean_ci(effect_values, samples=bootstrap_samples)
    effect_strength = _clip01(abs(effect) / 0.5)
    stat_conf = _clip01(1 / (1 + max(0, hi - lo) / max(abs(mean), 0.05)))

    # stability: compare first half and second half event effects
    mid = df['week_id'].median()
    e1 = event_study(df[df['week_id'] <= mid], treatment_col, metric_col)
    e2 = event_study(df[df['week_id'] > mid], treatment_col, metric_col)
    v1 = float(e1['effect_mean'].iloc[0]) if len(e1) else 0.0
    v2 = float(e2['effect_mean'].iloc[0]) if len(e2) else 0.0
    temporal_stability = _clip01(1 - abs(v1-v2) / (abs(effect)+1e-6+1))
    placebo_fpr = float(pl['placebo_abs_ge_real_rate'].iloc[0])
    placebo_pass = _clip01(1 - placebo_fpr)
    business_impact = _clip01(n_events / max(len(df) * 0.1, 1))
    pretrend_abs_diff = float(did['pretrend_abs_diff'].iloc[0]) if 'pretrend_abs_diff' in did else 0.0
    pretrend_quality = _clip01(1 / (1 + pretrend_abs_diff / max(abs(df[metric_col].mean()), 1.0)))
    confidence = (
        weights.get('effect_size_strength', .25) * effect_strength +
        weights.get('statistical_confidence', .20) * stat_conf +
        weights.get('temporal_stability', .20) * temporal_stability +
        weights.get('placebo_pass_score', .20) * placebo_pass +
        weights.get('business_impact_weight', .10) * business_impact +
        weights.get('pretrend_quality', .05) * pretrend_quality
    )
    return {
        'driver': treatment_col,
        'event_effect_mean': effect,
        'did_effect': float(did['did_effect'].iloc[0]),
        'bootstrap_mean': mean,
        'bootstrap_ci_low': lo,
        'bootstrap_ci_high': hi,
        'bootstrap_samples': bootstrap_samples,
        'placebo_false_positive_rate': placebo_fpr,
        'placebo_iterations': int(pl['placebo_iterations'].iloc[0]) if 'placebo_iterations' in pl else placebo_iterations,
        'pretrend_abs_diff': pretrend_abs_diff,
        'effect_size_strength': effect_strength,
        'statistical_confidence': stat_conf,
        'temporal_stability': temporal_stability,
        'placebo_pass_score': placebo_pass,
        'business_impact_weight': business_impact,
        'pretrend_quality': pretrend_quality,
        'driver_confidence': float(confidence),
    }


def score_all_drivers(
    df: pd.DataFrame,
    drivers: list[str],
    weights: dict | None = None,
    placebo_iterations: int = 200,
    bootstrap_samples: int = 200,
) -> pd.DataFrame:
    return pd.DataFrame([
        score_driver(df, d, weights=weights, placebo_iterations=placebo_iterations, bootstrap_samples=bootstrap_samples)
        for d in drivers
    ]).sort_values('driver_confidence', ascending=False)
