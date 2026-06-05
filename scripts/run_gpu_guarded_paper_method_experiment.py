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

from run_gpu_residual_experiment import (
    PANEL_PATH,
    OUT_DIR,
    REPORT_DIR,
    prepare_supervised,
    split_by_time,
    wape,
    mae,
    bias,
)


class SmallMLP(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def metric_row(split: str, model: str, y: np.ndarray, pred: np.ndarray, alpha: float = 0.0) -> dict:
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


def add_period_feature(df: pd.DataFrame, week_col: str) -> pd.Series:
    vals = pd.to_numeric(df[week_col], errors="coerce")
    if vals.notna().any():
        return (vals.astype(int) % 52).astype(int)
    return pd.Series(pd.Categorical(df[week_col]).codes % 52, index=df.index)


def build_expected_candidates(train: pd.DataFrame, frame: pd.DataFrame, group_cols: list[str], week_col: str) -> dict[str, np.ndarray]:
    train = train.copy()
    frame = frame.copy()

    train["_period"] = add_period_feature(train, week_col)
    frame["_period"] = add_period_feature(frame, week_col)

    global_mean = float(train["target"].mean())

    group_key_train = list(map(tuple, train[group_cols].to_numpy()))
    group_key_frame = list(map(tuple, frame[group_cols].to_numpy()))

    train["_group_key"] = group_key_train
    frame["_group_key"] = group_key_frame

    group_mean = train.groupby("_group_key")["target"].mean().to_dict()
    period_mean = train.groupby("_period")["target"].mean().to_dict()

    frame_group = frame["_group_key"].map(group_mean).fillna(global_mean).astype(float).to_numpy()
    frame_period = frame["_period"].map(period_mean).fillna(global_mean).astype(float).to_numpy()
    frame_roll = frame["baseline_pred"].to_numpy(dtype=float)

    candidates = {
        "rolling_baseline": frame_roll,
        "group_mean_only": frame_group,
        "period_mean_only": frame_period,
    }

    for beta in [0.10, 0.20, 0.30, 0.40]:
        candidates[f"roll_group_blend_{beta:.2f}"] = np.maximum((1 - beta) * frame_roll + beta * frame_group, 0.0)
        candidates[f"roll_period_blend_{beta:.2f}"] = np.maximum((1 - beta) * frame_roll + beta * frame_period, 0.0)
        candidates[f"roll_group_period_blend_{beta:.2f}"] = np.maximum(
            (1 - beta) * frame_roll + 0.5 * beta * frame_group + 0.5 * beta * frame_period,
            0.0,
        )

    return candidates


def add_dht_proxy(train: pd.DataFrame, frames: list[pd.DataFrame], group_cols: list[str]) -> list[np.ndarray]:
    # Safer PHT-inspired proxy: sequence-position plus normalized local demand state.
    outs = []
    for frame in frames:
        f = frame.copy()
        f["_pos"] = f.groupby(group_cols).cumcount()
        f["_max_pos"] = f.groupby(group_cols)["_pos"].transform("max").replace(0, 1)
        raw_progress = (f["_pos"] / f["_max_pos"]).clip(0, 1)

        local_level = f["rolling_mean_4"].fillna(f["baseline_pred"]).astype(float)
        denom = f.groupby(group_cols)["target"].transform("mean").replace(0, np.nan)
        rel_level = (local_level / denom).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 2) / 2.0

        outs.append((0.5 * raw_progress + 0.5 * rel_level).to_numpy(dtype=float))
    return outs


def train_residual_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], device: torch.device):
    x_mean = train[feature_cols].mean()
    x_std = train[feature_cols].std().replace(0, 1.0).fillna(1.0)

    y = train["residual_target"].to_numpy(dtype=float)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y)) if float(np.std(y)) > 1e-8 else 1.0

    def tx(frame: pd.DataFrame) -> np.ndarray:
        return ((frame[feature_cols] - x_mean) / x_std).fillna(0.0).to_numpy(dtype=np.float32)

    x_train = tx(train)
    y_train = ((y - y_mean) / y_std).astype(np.float32)
    x_val = tx(val)
    x_test = tx(test)

    model = SmallMLP(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    dl = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=256, shuffle=True)

    start = time.time()
    model.train()
    for _ in range(80):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    def pred(x: np.ndarray) -> np.ndarray:
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.from_numpy(x[i:i + 4096]).to(device)
                out.append(model(xb).detach().cpu().numpy())
        return (np.concatenate(out) if out else np.array([], dtype=float)) * y_std + y_mean

    return pred(x_val), pred(x_test), {"device": str(device), "train_seconds": time.time() - start}


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError("Run python -m src.main --project all first.")

    torch.manual_seed(42)
    np.random.seed(42)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH)
    supervised, feature_cols, week_col, group_cols = prepare_supervised(panel)
    train, val, test = split_by_time(supervised, week_col)

    y_train = train["target"].to_numpy(dtype=float)
    y_val = val["target"].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)

    val_candidates = build_expected_candidates(train, val, group_cols, week_col)
    test_candidates = build_expected_candidates(train, test, group_cols, week_col)
    train_candidates = build_expected_candidates(train, train, group_cols, week_col)

    candidate_rows = []
    for name, pred in val_candidates.items():
        candidate_rows.append(metric_row("validation", f"expected_candidate__{name}", y_val, pred))
    candidate_df = pd.DataFrame(candidate_rows)

    baseline_val_wape = wape(y_val, val_candidates["rolling_baseline"])

    best_expected_name = str(candidate_df.sort_values(["wape", "abs_bias"]).iloc[0]["model"]).replace("expected_candidate__", "")
    best_expected_val_wape = float(candidate_df.sort_values(["wape", "abs_bias"]).iloc[0]["wape"])

    # Critical guard: never allow a structural expected path worse than rolling baseline.
    if best_expected_val_wape <= baseline_val_wape:
        selected_base_name = best_expected_name
    else:
        selected_base_name = "rolling_baseline"

    train2 = train.copy()
    val2 = val.copy()
    test2 = test.copy()

    train2["selected_expected_path"] = train_candidates[selected_base_name]
    val2["selected_expected_path"] = val_candidates[selected_base_name]
    test2["selected_expected_path"] = test_candidates[selected_base_name]

    dht_train, dht_val, dht_test = add_dht_proxy(train2, [train2, val2, test2], group_cols)
    train2["demand_health_timestep"] = dht_train
    val2["demand_health_timestep"] = dht_val
    test2["demand_health_timestep"] = dht_test

    train2["residual_target"] = y_train - train2["selected_expected_path"].to_numpy(dtype=float)

    upgraded_features = feature_cols + ["selected_expected_path", "demand_health_timestep"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_resid, test_resid, meta = train_residual_model(train2, val2, test2, upgraded_features, device)

    base_val = val2["selected_expected_path"].to_numpy(dtype=float)
    base_test = test2["selected_expected_path"].to_numpy(dtype=float)

    alpha_rows = []
    for alpha in np.round(np.linspace(0.0, 1.0, 11), 2):
        val_pred = np.maximum(base_val + alpha * val_resid, 0.0)
        test_pred = np.maximum(base_test + alpha * test_resid, 0.0)
        alpha_rows.append(metric_row("validation", "guarded_paper_alpha_grid", y_val, val_pred, float(alpha)))
        alpha_rows.append(metric_row("test", "guarded_paper_alpha_grid", y_test, test_pred, float(alpha)))

    alpha_df = pd.DataFrame(alpha_rows)
    val_grid = alpha_df[alpha_df["split"] == "validation"].copy()

    rolling_val = val_candidates["rolling_baseline"]
    rolling_test = test_candidates["rolling_baseline"]
    rolling_val_wape = wape(y_val, rolling_val)

    eligible = val_grid[
        (val_grid["wape"] <= rolling_val_wape * 0.995)
        & (val_grid["alpha"] > 0.0)
    ].copy()

    # Conservative reliability gate:
    # choose the smallest validation-improving alpha, not the most aggressive
    # validation-WAPE winner. This prevents over-correction under test shift.
    if len(eligible):
        selected_alpha = float(eligible.sort_values(["alpha", "wape"]).iloc[0]["alpha"])
    else:
        selected_alpha = 0.0

    val_selected = np.maximum(base_val + selected_alpha * val_resid, 0.0)
    test_selected = np.maximum(base_test + selected_alpha * test_resid, 0.0)

    metrics = pd.DataFrame([
        metric_row("validation", "rolling_baseline", y_val, rolling_val, 0.0),
        metric_row("test", "rolling_baseline", y_test, rolling_test, 0.0),
        metric_row("validation", f"selected_expected_path__{selected_base_name}", y_val, base_val, 0.0),
        metric_row("test", f"selected_expected_path__{selected_base_name}", y_test, base_test, 0.0),
        metric_row("validation", "guarded_pht_expected_gpu_bias", y_val, val_selected, selected_alpha),
        metric_row("test", "guarded_pht_expected_gpu_bias", y_test, test_selected, selected_alpha),
    ])

    candidate_path = OUT_DIR / "guarded_paper_expected_path_candidates.csv"
    alpha_path = OUT_DIR / "guarded_paper_alpha_grid.csv"
    metrics_path = OUT_DIR / "guarded_paper_method_metrics.csv"
    report_path = REPORT_DIR / "guarded_paper_method_report.md"

    candidate_df.to_csv(candidate_path, index=False)
    alpha_df.to_csv(alpha_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    base_test_row = metrics[(metrics["split"] == "test") & (metrics["model"] == "rolling_baseline")].iloc[0]
    sel_test_row = metrics[(metrics["split"] == "test") & (metrics["model"] == "guarded_pht_expected_gpu_bias")].iloc[0]

    lines = [
        "# Guarded Paper-Method Upgrade Report",
        "",
        "## Diagnosis fixed",
        "This version prevents a paper-inspired expected path or neural correction from replacing the rolling baseline unless validation metrics justify it.",
        "",
        "## Selection",
        f"- Selected expected path: {selected_base_name}",
        f"- Best expected path validation WAPE: {best_expected_val_wape:.6f}",
        f"- Rolling baseline validation WAPE: {rolling_val_wape:.6f}",
        f"- Selected residual alpha: {selected_alpha}",
        f"- Device: {meta['device']}",
        f"- Training seconds: {meta['train_seconds']:.2f}",
        "",
        "## Test comparison",
        f"- Rolling baseline test WAPE: {float(base_test_row['wape']):.6f}",
        f"- Guarded paper-method test WAPE: {float(sel_test_row['wape']):.6f}",
        f"- Rolling baseline test abs bias: {float(base_test_row['abs_bias']):.6f}",
        f"- Guarded paper-method test abs bias: {float(sel_test_row['abs_bias']):.6f}",
        "",
        "## Metrics",
        metrics.to_markdown(index=False),
        "",
        "## Expected-path validation candidates",
        candidate_df.sort_values(['wape', 'abs_bias']).to_markdown(index=False),
        "",
        "## Claim boundary",
        "This is a guarded paper-method-inspired adaptation. It should be claimed only if validation-selected test metrics improve or remain safely bounded.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {candidate_path}")
    print(f"Wrote: {alpha_path}")
    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {report_path}")
    print("Selected expected path:", selected_base_name)
    print("Selected alpha:", selected_alpha)
    print("Test baseline:", base_test_row.to_dict())
    print("Test guarded:", sel_test_row.to_dict())


if __name__ == "__main__":
    main()
