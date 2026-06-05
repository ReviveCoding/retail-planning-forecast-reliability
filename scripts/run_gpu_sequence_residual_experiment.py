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


class LSTMResidual(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class AttentionResidual(nn.Module):
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
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos[:, : x.shape[1], :]
        z = self.encoder(h)
        pooled = z.mean(dim=1)
        return self.head(pooled).squeeze(-1)


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


def build_sequence_arrays(
    supervised: pd.DataFrame,
    feature_cols: list[str],
    week_col: str,
    group_cols: list[str],
    train_idx: set[int],
    val_idx: set[int],
    test_idx: set[int],
    lookback: int = 8,
) -> dict:
    df = supervised.copy()
    df["_row_id"] = df.index
    df = df.sort_values(group_cols + [week_col]).reset_index(drop=True)

    x_mean = supervised.loc[list(train_idx), feature_cols].mean()
    x_std = supervised.loc[list(train_idx), feature_cols].std().replace(0, 1.0).fillna(1.0)

    split_data = {
        "train": {"x": [], "y": [], "target": [], "baseline": [], "groups": []},
        "validation": {"x": [], "y": [], "target": [], "baseline": [], "groups": []},
        "test": {"x": [], "y": [], "target": [], "baseline": [], "groups": []},
    }

    for _, group in df.groupby(group_cols, sort=False):
        group = group.sort_values(week_col).reset_index(drop=True)
        x_group = ((group[feature_cols] - x_mean) / x_std).fillna(0.0).to_numpy(dtype=np.float32)

        for i in range(lookback - 1, len(group)):
            row_id = int(group.loc[i, "_row_id"])
            if row_id in train_idx:
                split = "train"
            elif row_id in val_idx:
                split = "validation"
            elif row_id in test_idx:
                split = "test"
            else:
                continue

            seq = x_group[i - lookback + 1 : i + 1]
            split_data[split]["x"].append(seq)
            split_data[split]["y"].append(float(group.loc[i, "residual_target"]))
            split_data[split]["target"].append(float(group.loc[i, "target"]))
            split_data[split]["baseline"].append(float(group.loc[i, "baseline_pred"]))
            split_data[split]["groups"].append(tuple(group.loc[i, group_cols].to_list()))

    for split in split_data:
        if split_data[split]["x"]:
            split_data[split]["x"] = np.stack(split_data[split]["x"]).astype(np.float32)
        else:
            split_data[split]["x"] = np.empty((0, lookback, len(feature_cols)), dtype=np.float32)
        for key in ["y", "target", "baseline"]:
            split_data[split][key] = np.array(split_data[split][key], dtype=np.float32)

    return split_data


def train_sequence_model(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    device: torch.device,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    train_y_scaled = ((train_y - y_mean) / y_std).astype(np.float32)

    if model_name == "gpu_lstm_residual":
        model = LSTMResidual(n_features=train_x.shape[-1]).to(device)
    elif model_name == "gpu_attention_residual":
        model = AttentionResidual(n_features=train_x.shape[-1], lookback=lookback).to(device)
    else:
        raise ValueError(model_name)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y_scaled))
    dl = DataLoader(ds, batch_size=256, shuffle=True)

    start = time.time()
    model.train()
    for _epoch in range(100):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
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
            return np.array([], dtype=float)
        pred_scaled = np.concatenate(preds)
        return pred_scaled * y_std + y_mean

    val_resid = predict(val_x)
    test_resid = predict(test_x)

    meta = {
        "model": model_name,
        "device": str(device),
        "train_seconds": train_seconds,
        "n_train": int(len(train_x)),
        "n_val": int(len(val_x)),
        "n_test": int(len(test_x)),
        "lookback": int(lookback),
        "n_features": int(train_x.shape[-1]),
    }
    return val_resid, test_resid, meta


def select_alphas(
    val_y: np.ndarray,
    val_base: np.ndarray,
    val_resid: np.ndarray,
) -> dict[str, float]:
    base_wape = wape(val_y, val_base)
    rows = []
    for alpha in np.round(np.linspace(0.0, 1.0, 11), 2):
        pred = np.maximum(val_base + alpha * val_resid, 0.0)
        rows.append(
            {
                "alpha": float(alpha),
                "wape": wape(val_y, pred),
                "abs_bias": abs(bias(val_y, pred)),
            }
        )
    grid = pd.DataFrame(rows)

    alpha_best = float(grid.sort_values(["wape", "abs_bias"]).iloc[0]["alpha"])

    eligible = grid[grid["wape"] <= base_wape * 0.99].copy()
    alpha_conservative = float(eligible.sort_values(["alpha", "wape"]).iloc[0]["alpha"]) if len(eligible) else 0.0

    bias_eligible = grid[grid["wape"] <= base_wape * 1.01].copy()
    alpha_bias = float(bias_eligible.sort_values(["abs_bias", "wape"]).iloc[0]["alpha"]) if len(bias_eligible) else 0.0

    return {
        "baseline_alpha_0": 0.0,
        "validation_wape_best": alpha_best,
        "conservative_wape_gate": alpha_conservative,
        "bias_aware_gate": alpha_bias,
    }


def high_bias_mask_from_train(
    train_frame: pd.DataFrame,
    test_groups: list[tuple],
    group_cols: list[str],
) -> np.ndarray:
    train_bias = train_frame.copy()
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
    if not len(bias_by_group):
        return np.ones(len(test_groups), dtype=bool)

    threshold = bias_by_group["abs_mean_bias"].quantile(0.8)
    high_bias_groups = bias_by_group[bias_by_group["abs_mean_bias"] >= threshold][group_cols]
    high_bias_key = set(map(tuple, high_bias_groups.to_numpy()))
    return np.array([tuple(g) in high_bias_key for g in test_groups])


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

    all_rows = []
    high_bias_rows = []
    meta_rows = []

    val_y = seq["validation"]["target"]
    val_base = seq["validation"]["baseline"]
    test_y = seq["test"]["target"]
    test_base = seq["test"]["baseline"]

    hb_mask = high_bias_mask_from_train(
        train_frame=train_frame,
        test_groups=seq["test"]["groups"],
        group_cols=group_cols,
    )

    for model_name in ["gpu_lstm_residual", "gpu_attention_residual"]:
        val_resid, test_resid, meta = train_sequence_model(
            model_name=model_name,
            train_x=seq["train"]["x"],
            train_y=seq["train"]["y"],
            val_x=seq["validation"]["x"],
            test_x=seq["test"]["x"],
            device=device,
            lookback=lookback,
        )
        meta_rows.append(meta)

        policies = select_alphas(val_y, val_base, val_resid)

        for policy, alpha in policies.items():
            val_pred = np.maximum(val_base + alpha * val_resid, 0.0)
            test_pred = np.maximum(test_base + alpha * test_resid, 0.0)

            all_rows.append(metric_row("validation", f"{model_name}__{policy}", val_y, val_pred, alpha))
            all_rows.append(metric_row("test", f"{model_name}__{policy}", test_y, test_pred, alpha))

            if hb_mask.sum() > 0:
                hb_y = test_y[hb_mask]
                hb_base = test_base[hb_mask]
                hb_resid = test_resid[hb_mask]
                hb_pred = np.maximum(hb_base + alpha * hb_resid, 0.0)
                high_bias_rows.append(
                    {
                        "slice": "high_bias_groups",
                        "model": f"{model_name}__{policy}",
                        "alpha": alpha,
                        "rows": int(hb_mask.sum()),
                        "wape": wape(hb_y, hb_pred),
                        "mae": mae(hb_y, hb_pred),
                        "bias": bias(hb_y, hb_pred),
                        "abs_bias": abs(bias(hb_y, hb_pred)),
                        "under_forecast_rate": float(np.mean(hb_pred < hb_y)),
                    }
                )

    baseline_rows = [
        metric_row("validation", "baseline_rolling_mean", val_y, val_base, 0.0),
        metric_row("test", "baseline_rolling_mean", test_y, test_base, 0.0),
    ]
    all_rows = baseline_rows + all_rows

    if hb_mask.sum() > 0:
        hb_y = test_y[hb_mask]
        hb_base = test_base[hb_mask]
        high_bias_rows = [
            {
                "slice": "high_bias_groups",
                "model": "baseline_rolling_mean",
                "alpha": 0.0,
                "rows": int(hb_mask.sum()),
                "wape": wape(hb_y, hb_base),
                "mae": mae(hb_y, hb_base),
                "bias": bias(hb_y, hb_base),
                "abs_bias": abs(bias(hb_y, hb_base)),
                "under_forecast_rate": float(np.mean(hb_base < hb_y)),
            }
        ] + high_bias_rows

    metrics = pd.DataFrame(all_rows)
    high_bias = pd.DataFrame(high_bias_rows)
    meta_df = pd.DataFrame(meta_rows)

    metrics_path = OUT_DIR / "gpu_sequence_residual_metrics.csv"
    high_bias_path = OUT_DIR / "gpu_sequence_residual_high_bias_metrics.csv"
    meta_path = OUT_DIR / "gpu_sequence_residual_meta.csv"
    report_path = REPORT_DIR / "gpu_sequence_residual_report.md"

    metrics.to_csv(metrics_path, index=False)
    high_bias.to_csv(high_bias_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    test_metrics = metrics[metrics["split"] == "test"].copy()
    best_test = test_metrics.sort_values(["wape", "abs_bias"]).iloc[0]

    lines = [
        "# GPU Sequence Residual Correction Report",
        "",
        "## Experiment",
        "This experiment compares GPU LSTM and tiny attention residual-correction models against the rolling baseline.",
        "Residual corrections are applied only through validation-selected alpha policies.",
        "",
        "## GPU/model metadata",
        meta_df.to_markdown(index=False),
        "",
        "## Best test policy",
        f"- Model: {best_test['model']}",
        f"- Alpha: {float(best_test['alpha']):.2f}",
        f"- Test WAPE: {float(best_test['wape']):.6f}",
        f"- Test MAE: {float(best_test['mae']):.6f}",
        f"- Test abs bias: {float(best_test['abs_bias']):.6f}",
        "",
        "## All metrics",
        metrics.to_markdown(index=False),
        "",
        "## High-bias slice metrics",
        high_bias.to_markdown(index=False) if len(high_bias) else "No high-bias rows available.",
        "",
        "## Claim boundary",
        "Use test metrics only for reporting, not for model selection.",
        "Selection policies use validation metrics only.",
        "This is an optional GPU experiment using public/sample retail panels.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {high_bias_path}")
    print(f"Wrote: {meta_path}")
    print(f"Wrote: {report_path}")
    print("Best test policy:", best_test.to_dict())


if __name__ == "__main__":
    main()
