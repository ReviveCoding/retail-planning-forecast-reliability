from __future__ import annotations
import numpy as np
import pandas as pd


def max_identity_error(df: pd.DataFrame) -> float:
    if 'identity_error' not in df:
        return float('inf')
    return float(np.abs(df['identity_error']).max())


def identity_passes(df: pd.DataFrame, tolerance=1e-6) -> bool:
    return max_identity_error(df) <= tolerance
