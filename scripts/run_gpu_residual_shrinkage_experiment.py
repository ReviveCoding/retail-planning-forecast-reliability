from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from run_gpu_residual_experiment import (  # noqa: E402
    PANEL_PATH,
    OUT_DIR,
    REPORT_DIR,
    prepare_supervised,
    split_by_time,
    train_residual_model,
    wape,
    mae,
    bias,
)


def metric_row(split: str, model: str, y: np.ndarray, pred: np.ndarray, alpha: float) -> dict:
    return {
        "split": split,
        "model": model,
        "alpha": alpha,
        "wape": wape(y, pred),
        "mae": mae(y, pred),
        "bias": bias(y, pred),
        "abs_bias": abs(bias(y, pred)),
        "under_forecast_rate": float(np.mean(pred < y)),
    }


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found. Run: python -m src.main --project all"
        )

    torch.manual_seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    supervised, feature_cols, week_col, group_cols = prepare_supervised(panel)
    train, val, test = split_by_time(supervised, week_col)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_resid_pred, test_resid_pred, meta = train_residual_model(
        train=train,
        val=val,
        test=test,
        feature_cols=feature_cols,
        device=device,
    )

    val_y = val["target"].to_numpy(dtype=float)
    test_y = test["target"].to_numpy(dtype=float)

    val_base = val["baseline_pred"].to_numpy(dtype=float)
    test_base = test["baseline_pred"].to_numpy(dtype=float)

    base_val_wape = wape(val_y, val_base)
    base_val_abs_bias = abs(bias(val_y, val_base))

    alpha_grid = np.round(np.linspace(0.0, 1.0, 11), 2)

    rows = []
    for alpha in alpha_grid:
        val_pred = np.maximum(val_base + alpha * val_resid_pred, 0.0)
        test_pred = np.maximum(test_base + alpha * test_resid_pred, 0.0)

        rows.append(metric_row("validation", "gpu_residual_alpha_grid", val_y, val_pred, float(alpha)))
        rows.append(metric_row("test", "gpu_residual_alpha_grid", test_y, test_pred, float(alpha)))

    grid = pd.DataFrame(rows)
    val_grid = grid[grid["split"] == "validation"].copy()

    # Policy 1: pure validation WAPE minimization.
    alpha_wape = float(val_grid.sort_values(["wape", "abs_bias"]).iloc[0]["alpha"])

    # Policy 2: conservative shrinkage gate.
    # Only use corrections that improve validation WAPE by at least 1%.
    # Among those, prefer smaller alpha to reduce over-correction risk.
    eligible = val_grid[val_grid["wape"] <= base_val_wape * 0.99].copy()
    if len(eligible):
        alpha_conservative = float(eligible.sort_values(["alpha", "wape"]).iloc[0]["alpha"])
    else:
        alpha_conservative = 0.0

    # Policy 3: bias-aware gate within a 1% validation WAPE tolerance.
    # This is useful when the business goal is reducing systematic under/over forecast.
    bias_eligible = val_grid[val_grid["wape"] <= base_val_wape * 1.01].copy()
    if len(bias_eligible):
        alpha_bias_aware = float(bias_eligible.sort_values(["abs_bias", "wape"]).iloc[0]["alpha"])
    else:
        alpha_bias_aware = 0.0

    selected_policies = {
        "baseline_alpha_0": 0.0,
        "validation_wape_best": alpha_wape,
        "conservative_wape_gate": alpha_conservative,
        "bias_aware_gate": alpha_bias_aware,
    }

    selected_rows = []
    for name, alpha in selected_policies.items():
        val_pred = np.maximum(val_base + alpha * val_resid_pred, 0.0)
        test_pred = np.maximum(test_base + alpha * test_resid_pred, 0.0)

        selected_rows.append(metric_row("validation", name, val_y, val_pred, alpha))
        selected_rows.append(metric_row("test", name, test_y, test_pred, alpha))

    selected = pd.DataFrame(selected_rows)

    # High-bias slice based only on train-period baseline behavior.
    train_bias = train.copy()
    train_bias["baseline_error"] = train_bias["target"] - train_bias["baseline_pred"]
    bias_by_group = (
        train_bias.groupby(group_cols)
        .agg(
            n=("baseline_error", "size"),
            abs_mean_bias=("baseline_error", lambda s: float(abs(np.mean(s)))),
        )
        .reset_index()
    )
    bias_by_group = bias_by_group[bias_by_group["n"] >= 3]

    high_bias_rows = []
    if len(bias_by_group):
        threshold = bias_by_group["abs_mean_bias"].quantile(0.8)
        high_bias_groups = bias_by_group[bias_by_group["abs_mean_bias"] >= threshold][group_cols]
        high_bias_key = set(map(tuple, high_bias_groups.to_numpy()))
        test_keys = list(map(tuple, test[group_cols].to_numpy()))
        high_bias_mask = np.array([k in high_bias_key for k in test_keys])

        if high_bias_mask.sum() > 0:
            hb_y = test_y[high_bias_mask]
            hb_base = test_base[high_bias_mask]
            hb_resid = test_resid_pred[high_bias_mask]

            for name, alpha in selected_policies.items():
                hb_pred = np.maximum(hb_base + alpha * hb_resid, 0.0)
                high_bias_rows.append(
                    {
                        "slice": "high_bias_groups",
                        "model": name,
                        "alpha": alpha,
                        "rows": int(high_bias_mask.sum()),
                        "wape": wape(hb_y, hb_pred),
                        "mae": mae(hb_y, hb_pred),
                        "bias": bias(hb_y, hb_pred),
                        "abs_bias": abs(bias(hb_y, hb_pred)),
                        "under_forecast_rate": float(np.mean(hb_pred < hb_y)),
                    }
                )

    high_bias = pd.DataFrame(high_bias_rows)

    grid_path = OUT_DIR / "gpu_residual_alpha_grid_metrics.csv"
    selected_path = OUT_DIR / "gpu_residual_shrinkage_selected_metrics.csv"
    high_bias_path = OUT_DIR / "gpu_residual_shrinkage_high_bias_metrics.csv"
    report_path = REPORT_DIR / "gpu_residual_shrinkage_report.md"

    grid.to_csv(grid_path, index=False)
    selected.to_csv(selected_path, index=False)
    high_bias.to_csv(high_bias_path, index=False)

    lines = [
        "# GPU Residual Shrinkage Gate Report",
        "",
        "## GPU status",
        f"- Device: {meta['device']}",
        f"- Train rows: {meta['n_train']}",
        f"- Validation rows: {meta['n_val']}",
        f"- Test rows: {meta['n_test']}",
        f"- Features: {meta['n_features']}",
        f"- Training seconds: {meta['train_seconds']:.2f}",
        "",
        "## Why this experiment exists",
        "The first GPU residual correction applied the full predicted residual when validation WAPE improved.",
        "That improved validation WAPE but slightly worsened test WAPE, suggesting over-correction.",
        "This experiment adds validation-selected alpha shrinkage to reduce over-correction risk.",
        "",
        "## Selected alpha policies",
        f"- validation_wape_best alpha: {alpha_wape}",
        f"- conservative_wape_gate alpha: {alpha_conservative}",
        f"- bias_aware_gate alpha: {alpha_bias_aware}",
        "",
        "## Selected metrics",
        selected.to_markdown(index=False),
        "",
        "## High-bias slice metrics",
        high_bias.to_markdown(index=False) if len(high_bias) else "No high-bias slice rows available.",
        "",
        "## Claim boundary",
        "Use test metrics only for reporting, not for model selection.",
        "Selection policies use validation metrics only.",
        "This is an optional GPU experiment using public/sample retail panels.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {grid_path}")
    print(f"Wrote: {selected_path}")
    print(f"Wrote: {high_bias_path}")
    print(f"Wrote: {report_path}")
    print("Selected alphas:", selected_policies)


if __name__ == "__main__":
    main()
