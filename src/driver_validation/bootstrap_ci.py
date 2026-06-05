from __future__ import annotations
import numpy as np


def bootstrap_mean_ci(values, samples=200, seed=42, alpha=0.05) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(samples)]
    return float(arr.mean()), float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))
