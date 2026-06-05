from __future__ import annotations
import pandas as pd
import numpy as np


def compute_asymmetric_coverage_metrics(df: pd.DataFrame, forecast_col='units_forecast', actual_col='target_units', buffer_factor=1.10, correction_factor=0.50) -> pd.DataFrame:
    out = df.copy()
    out['buffered_forecast'] = out[forecast_col] * buffer_factor
    # Bias correction reduces buffer when recent forecast overreach is likely.
    recent_overreach = (out.get('rolling_mean_4', out[forecast_col]) - out[actual_col]).clip(lower=0)
    out['corrected_buffered_forecast'] = (out['buffered_forecast'] - correction_factor * recent_overreach).clip(lower=0)
    for col, prefix in [('buffered_forecast','buffer_only'), ('corrected_buffered_forecast','buffer_corrected')]:
        covered = np.minimum(out[col], out[actual_col])
        out[f'{prefix}_coverage_recall'] = covered / out[actual_col].clip(lower=1)
        out[f'{prefix}_buffer_precision'] = covered / out[col].clip(lower=1)
        rec = out[f'{prefix}_coverage_recall']
        prec = out[f'{prefix}_buffer_precision']
        out[f'{prefix}_coverage_f1'] = 2 * rec * prec / (rec + prec).replace(0, np.nan)
        out[f'{prefix}_coverage_f1'] = out[f'{prefix}_coverage_f1'].fillna(0)
    return out


def summarize_coverage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for prefix in ['buffer_only', 'buffer_corrected']:
        rows.append({
            'approach': prefix,
            'coverage_recall': df[f'{prefix}_coverage_recall'].mean(),
            'buffer_precision': df[f'{prefix}_buffer_precision'].mean(),
            'coverage_f1': df[f'{prefix}_coverage_f1'].mean(),
        })
    return pd.DataFrame(rows)
