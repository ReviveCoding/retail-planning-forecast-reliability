from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "scripts"))

from run_gpu_residual_experiment import (  # noqa: E402
    PANEL_PATH,
    OUT_DIR,
    REPORT_DIR,
    prepare_supervised,
    split_by_time,
    wape,
    mae,
    bias,
)
from run_gpu_sequence_residual_experiment import (  # noqa: E402
    build_sequence_arrays,
    high_bias_mask_from_train,
)


QUANTILES = [0.50, 0.80, 0.90]


class QuantileAttentionResidual(nn.Module):
    def __init__(self, n_features: int, lookback: int, d_model: int = 32, nhead: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, lookback, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=96,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(64, 3),
        )
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        raw = self.head(pooled)

        q50 = raw[:, 0]
        q80 = q50 + self.softplus(raw[:, 1])
        q90 = q80 + self.softplus(raw[:, 2])

        return torch.stack([q50, q80, q90], dim=1)


def pinball_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = []
    for i, q in enumerate(QUANTILES):
        err = target - pred[:, i]
        losses.append(torch.maximum(q * err, (q - 1.0) * err).mean())
    return torch.stack(losses).mean()


def coverage_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 0.0)

    covered = np.minimum(y, pred)
    demand_sum = float(np.sum(y))
    pred_sum = float(np.sum(pred))

    recall = float(np.sum(covered) / demand_sum) if demand_sum > 1e-12 else float("nan")
    precision = float(np.sum(covered) / pred_sum) if pred_sum > 1e-12 else float("nan")

    if np.isfinite(recall) and np.isfinite(precision) and (recall + precision) > 1e-12:
        f1 = float(2.0 * recall * precision / (recall + precision))
    else:
        f1 = float("nan")

    lost_sales_proxy = float(np.sum(np.maximum(y - pred, 0.0)) / demand_sum) if demand_sum > 1e-12 else float("nan")
    overstock_proxy = float(np.sum(np.maximum(pred - y, 0.0)) / demand_sum) if demand_sum > 1e-12 else float("nan")

    return {
        "coverage_recall": recall,
        "buffer_precision": precision,
        "coverage_f1": f1,
        "lost_sales_proxy": lost_sales_proxy,
        "overstock_proxy": overstock_proxy,
        "under_forecast_rate": float(np.mean(pred < y)),
        "wape": wape(y, pred),
        "mae": mae(y, pred),
        "bias": bias(y, pred),
        "abs_bias": abs(bias(y, pred)),
    }


def train_quantile_model(
    train_x: np.ndarray,
    train_residual_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_mean = float(train_residual_y.mean())
    y_std = float(train_residual_y.std())
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    y_scaled = ((train_residual_y - y_mean) / y_std).astype(np.float32)

    model = QuantileAttentionResidual(
        n_features=train_x.shape[-1],
        lookback=lookback,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(y_scaled))
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    start = time.time()
    model.train()
    for _epoch in range(120):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = pinball_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    train_seconds = time.time() - start

    def predict(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 2048):
                xb = torch.from_numpy(x[i : i + 2048]).to(device)
                pred = model(xb).detach().cpu().numpy()
                preds.append(pred)
        if not preds:
            return np.empty((0, 3), dtype=float)
        pred_scaled = np.concatenate(preds, axis=0)
        return pred_scaled * y_std + y_mean

    val_resid_quantiles = predict(val_x)
    test_resid_quantiles = predict(test_x)

    meta = {
        "model": "gpu_attention_quantile_residual",
        "device": str(device),
        "train_seconds": train_seconds,
        "n_train": int(len(train_x)),
        "n_val": int(len(val_x)),
        "n_test": int(len(test_x)),
        "lookback": int(lookback),
        "n_features": int(train_x.shape[-1]),
        "quantiles": ",".join(str(q) for q in QUANTILES),
    }

    return val_resid_quantiles, test_resid_quantiles, meta


def evaluate_candidate(
    split: str,
    model: str,
    y: np.ndarray,
    pred: np.ndarray,
    policy_group: str,
) -> dict:
    out = {
        "split": split,
        "model": model,
        "policy_group": policy_group,
    }
    out.update(coverage_metrics(y, pred))
    return out


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"{PANEL_PATH} not found. Run: python -m src.main --project all")

    torch.manual_seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lookback = 8
    panel = pd.read_csv(PANEL_PATH)
    supervised, feature_cols, week_col, group_cols = prepare_supervised(panel)
    train_frame, val_frame, test_frame = split_by_time(supervised, week_col)

    seq = build_sequence_arrays(
        supervised=supervised,
        feature_cols=feature_cols,
        week_col=week_col,
        group_cols=group_cols,
        train_idx=set(train_frame.index),
        val_idx=set(val_frame.index),
        test_idx=set(test_frame.index),
        lookback=lookback,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_resid_q, test_resid_q, meta = train_quantile_model(
        train_x=seq["train"]["x"],
        train_residual_y=seq["train"]["y"],
        val_x=seq["validation"]["x"],
        test_x=seq["test"]["x"],
        device=device,
        lookback=lookback,
    )

    val_y = seq["validation"]["target"].astype(float)
    val_base = seq["validation"]["baseline"].astype(float)
    test_y = seq["test"]["target"].astype(float)
    test_base = seq["test"]["baseline"].astype(float)

    rows = []

    fixed_buffers = {
        "baseline_point": 1.00,
        "fixed_buffer_05pct": 1.05,
        "fixed_buffer_10pct": 1.10,
        "fixed_buffer_15pct": 1.15,
    }

    for name, mult in fixed_buffers.items():
        rows.append(evaluate_candidate("validation", name, val_y, val_base * mult, "fixed_buffer"))
        rows.append(evaluate_candidate("test", name, test_y, test_base * mult, "fixed_buffer"))

    for idx, q in enumerate(QUANTILES):
        val_pred = np.maximum(val_base + val_resid_q[:, idx], 0.0)
        test_pred = np.maximum(test_base + test_resid_q[:, idx], 0.0)
        name = f"gpu_quantile_q{int(q * 100)}"
        rows.append(evaluate_candidate("validation", name, val_y, val_pred, "gpu_quantile"))
        rows.append(evaluate_candidate("test", name, test_y, test_pred, "gpu_quantile"))

    metrics = pd.DataFrame(rows)

    val_candidates = metrics[metrics["split"] == "validation"].copy()

    # Planning policy:
    # maximize coverage_f1 first, but reject candidates with extreme overstock.
    baseline_val = val_candidates[val_candidates["model"] == "baseline_point"].iloc[0]
    max_allowed_overstock = max(float(baseline_val["overstock_proxy"]) + 0.05, 0.10)

    eligible = val_candidates[val_candidates["overstock_proxy"] <= max_allowed_overstock].copy()
    if not len(eligible):
        eligible = val_candidates.copy()

    selected = eligible.sort_values(
        ["coverage_f1", "lost_sales_proxy", "buffer_precision"],
        ascending=[False, True, False],
    ).iloc[0]

    selected_model = str(selected["model"])
    selected_test = metrics[(metrics["split"] == "test") & (metrics["model"] == selected_model)].iloc[0]

    hb_mask = high_bias_mask_from_train(
        train_frame=train_frame,
        test_groups=seq["test"]["groups"],
        group_cols=group_cols,
    )

    high_bias_rows = []
    if hb_mask.sum() > 0:
        hb_y = test_y[hb_mask]
        hb_base = test_base[hb_mask]

        for name, mult in fixed_buffers.items():
            row = {
                "slice": "high_bias_groups",
                "model": name,
                "policy_group": "fixed_buffer",
                "rows": int(hb_mask.sum()),
            }
            row.update(coverage_metrics(hb_y, hb_base * mult))
            high_bias_rows.append(row)

        for idx, q in enumerate(QUANTILES):
            hb_pred = np.maximum(hb_base + test_resid_q[hb_mask, idx], 0.0)
            row = {
                "slice": "high_bias_groups",
                "model": f"gpu_quantile_q{int(q * 100)}",
                "policy_group": "gpu_quantile",
                "rows": int(hb_mask.sum()),
            }
            row.update(coverage_metrics(hb_y, hb_pred))
            high_bias_rows.append(row)

    high_bias = pd.DataFrame(high_bias_rows)

    metrics_path = OUT_DIR / "gpu_quantile_asymmetric_buffer_metrics.csv"
    high_bias_path = OUT_DIR / "gpu_quantile_asymmetric_buffer_high_bias_metrics.csv"
    meta_path = OUT_DIR / "gpu_quantile_asymmetric_buffer_meta.csv"
    report_path = REPORT_DIR / "gpu_quantile_asymmetric_buffer_report.md"

    metrics.to_csv(metrics_path, index=False)
    high_bias.to_csv(high_bias_path, index=False)
    pd.DataFrame([meta]).to_csv(meta_path, index=False)

    baseline_test = metrics[(metrics["split"] == "test") & (metrics["model"] == "baseline_point")].iloc[0]

    lines = [
        "# GPU Quantile / Asymmetric Buffer Report",
        "",
        "## Experiment",
        "This experiment trains a GPU attention quantile residual model and compares q50/q80/q90 planning buffers against fixed safety buffers.",
        "The goal is not only point-forecast WAPE, but planning reliability: demand coverage, buffer precision, coverage F1, lost-sales proxy, and overstock proxy.",
        "",
        "## GPU/model metadata",
        pd.DataFrame([meta]).to_markdown(index=False),
        "",
        "## Validation-selected planning policy",
        f"- Selected model: {selected_model}",
        f"- Validation coverage F1: {float(selected['coverage_f1']):.6f}",
        f"- Validation lost-sales proxy: {float(selected['lost_sales_proxy']):.6f}",
        f"- Validation overstock proxy: {float(selected['overstock_proxy']):.6f}",
        "",
        "## Test comparison against baseline point forecast",
        f"- Baseline test coverage F1: {float(baseline_test['coverage_f1']):.6f}",
        f"- Selected test coverage F1: {float(selected_test['coverage_f1']):.6f}",
        f"- Baseline test lost-sales proxy: {float(baseline_test['lost_sales_proxy']):.6f}",
        f"- Selected test lost-sales proxy: {float(selected_test['lost_sales_proxy']):.6f}",
        f"- Baseline test overstock proxy: {float(baseline_test['overstock_proxy']):.6f}",
        f"- Selected test overstock proxy: {float(selected_test['overstock_proxy']):.6f}",
        "",
        "## All metrics",
        metrics.to_markdown(index=False),
        "",
        "## High-bias slice metrics",
        high_bias.to_markdown(index=False) if len(high_bias) else "No high-bias rows available.",
        "",
        "## Claim boundary",
        "Use validation metrics only for policy selection.",
        "Use test metrics only for reporting.",
        "This is an optional GPU planning-reliability experiment using public/sample retail panels.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {high_bias_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {report_path}")
    print("Selected policy:", selected_model)
    print("Selected test:", selected_test.to_dict())


if __name__ == "__main__":
    main()
