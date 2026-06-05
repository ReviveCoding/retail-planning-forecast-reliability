from __future__ import annotations
import numpy as np
import pandas as pd
from .event_study import event_study


def placebo_test(df: pd.DataFrame, treatment_col: str, metric_col='units_actual', iterations=200, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    effects = []
    d = df.copy()
    p = float(d[treatment_col].mean()) if treatment_col in d else 0.1
    p = min(max(p, 0.02), 0.3)
    for _ in range(iterations):
        d['placebo_flag'] = rng.binomial(1, p, len(d))
        eff = event_study(d, 'placebo_flag', metric_col)
        effects.append(float(eff['effect_mean'].iloc[0]) if len(eff) else 0.0)
    real = event_study(df, treatment_col, metric_col)
    real_effect = float(real['effect_mean'].iloc[0]) if len(real) else 0.0
    fpr = float(np.mean(np.abs(effects) >= abs(real_effect))) if effects else 1.0
    return pd.DataFrame([{
        'driver': treatment_col,
        'real_effect': real_effect,
        'placebo_abs_ge_real_rate': fpr,
        'placebo_mean': float(np.mean(effects)) if effects else 0.0,
        'placebo_std': float(np.std(effects)) if effects else 0.0,
        'placebo_iterations': iterations,
    }])
