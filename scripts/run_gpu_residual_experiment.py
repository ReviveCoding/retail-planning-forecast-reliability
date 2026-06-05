from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "outputs" / "processed" / "m5_weekly_panel.csv"
OUT_DIR = ROOT / "outputs" / "specialty"
REPORT_DIR = ROOT / "reports"


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


class ResidualMLP(nn.Module):
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


def infer_columns(df: pd.DataFrame) -> tuple[str, str, str, list[str]]:
    unit_candidates = ["units_actual", "target_units", "unit_sales", "sales", "demand"]
    units_col = next((c for c in unit_candidates if c in df.columns), None)
    if units_col is None:
        units_col = next((c for c in df.columns if "unit" in c.lower() or "sale" in c.lower()), None)
    if units_col is None:
        raise ValueError(f"Could not infer units column from columns: {list(df.columns)}")

    week_col = next((c for c in ["week_id", "week", "date", "week_start"] if c in df.columns), None)
    if week_col is None:
        raise ValueError(f"Could not infer week/date column from columns: {list(df.columns)}")

    group_cols = [c for c in ["item_id", "store_id"] if c in df.columns]
    if not group_cols:
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        group_cols = object_cols[:2]
    if not group_cols:
        raise ValueError("Could not infer item/store grouping columns.")

    return units_col, week_col, group_cols[0], group_cols


def prepare_supervised(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str, list[str]]:
    units_col, week_col, item_col, group_cols = infer_columns(panel)

    df = panel.copy()
    df[units_col] = pd.to_numeric(df[units_col], errors="coerce")
    df = df.dropna(subset=[units_col]).copy()

    if week_col == "date" or "date" in week_col.lower():
        df[week_col] = pd.to_datetime(df[week_col], errors="coerce")
    df = df.sort_values(group_cols + [week_col]).reset_index(drop=True)

    g = df.groupby(group_cols, sort=False)

    for lag in [1, 2, 4]:
        df[f"lag_{lag}"] = g[units_col].shift(lag)

    df["rolling_mean_4"] = g[units_col].transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    df["rolling_std_4"] = g[units_col].transform(lambda s: s.shift(1).rolling(4, min_periods=2).std()).fillna(0.0)

    df["baseline_pred"] = df["rolling_mean_4"].fillna(df["lag_1"])
    df["demand_state_timestep"] = g.cumcount() / 52.0
    df["demand_state_timestep"] = df["demand_state_timestep"].clip(0.0, 1.5)

    price_col = next((c for c in ["price_actual", "sell_price", "price"] if c in df.columns), None)
    if price_col is not None:
        df["price_feature"] = pd.to_numeric(df[price_col], errors="coerce")
    else:
        df["price_feature"] = 1.0

    event_col = next((c for c in ["event_flag", "promotion_flag", "promo_flag"] if c in df.columns), None)
    if event_col is not None:
        df["event_feature"] = pd.to_numeric(df[event_col], errors="coerce").fillna(0.0)
    else:
        df["event_feature"] = 0.0

    for col in group_cols:
        df[f"{col}_code"] = pd.Categorical(df[col]).codes.astype(float)

    df = df.dropna(subset=["baseline_pred", "lag_1", "lag_2", "lag_4"]).copy()
    df["target"] = df[units_col].astype(float)
    df["residual_target"] = df["target"] - df["baseline_pred"]

    feature_cols = [
        "lag_1",
        "lag_2",
        "lag_4",
        "rolling_mean_4",
        "rolling_std_4",
        "demand_state_timestep",
        "price_feature",
        "event_feature",
    ] + [f"{c}_code" for c in group_cols]

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    return df, feature_cols, week_col, group_cols


def split_by_time(df: pd.DataFrame, week_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    weeks = np.array(sorted(df[week_col].dropna().unique()))
    if len(weeks) < 5:
        n = len(df)
        train = df.iloc[: int(n * 0.6)]
        val = df.iloc[int(n * 0.6): int(n * 0.8)]
        test = df.iloc[int(n * 0.8):]
        return train, val, test

    train_end = max(1, int(len(weeks) * 0.6))
    val_end = max(train_end + 1, int(len(weeks) * 0.8))

    train_weeks = set(weeks[:train_end])
    val_weeks = set(weeks[train_end:val_end])
    test_weeks = set(weeks[val_end:])

    train = df[df[week_col].isin(train_weeks)].copy()
    val = df[df[week_col].isin(val_weeks)].copy()
    test = df[df[week_col].isin(test_weeks)].copy()
    return train, val, test


def train_residual_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    x_mean = train[feature_cols].mean()
    x_std = train[feature_cols].std().replace(0, 1.0).fillna(1.0)

    y_mean = float(train["residual_target"].mean())
    y_std = float(train["residual_target"].std())
    if not np.isfinite(y_std) or y_std < 1e-8:
        y_std = 1.0

    def make_xy(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = ((frame[feature_cols] - x_mean) / x_std).fillna(0.0).to_numpy(dtype=np.float32)
        y = ((frame["residual_target"].to_numpy(dtype=np.float32) - y_mean) / y_std).astype(np.float32)
        return x, y

    x_train, y_train = make_xy(train)
    x_val, _ = make_xy(val)
    x_test, _ = make_xy(test)

    model = ResidualMLP(n_features=x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    dl = DataLoader(ds, batch_size=512, shuffle=True)

    start = time.time()
    model.train()
    for _epoch in range(80):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    train_seconds = time.time() - start

    def predict_residual(x: np.ndarray) -> np.ndarray:
        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                xb = torch.from_numpy(x[i:i + 4096]).to(device)
                pred = model(xb).detach().cpu().numpy()
                preds.append(pred)
        pred_scaled = np.concatenate(preds) if preds else np.array([], dtype=float)
        return pred_scaled * y_std + y_mean

    val_resid_pred = predict_residual(x_val)
    test_resid_pred = predict_residual(x_test)

    meta = {
        "device": str(device),
        "train_seconds": train_seconds,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
    }
    return val_resid_pred, test_resid_pred, meta


def main() -> None:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(
            f"{PANEL_PATH} not found. Run: python -m src.main --project all"
        )

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

    val_corrected = np.maximum(val_base + val_resid_pred, 0.0)
    test_corrected = np.maximum(test_base + test_resid_pred, 0.0)

    val_base_wape = wape(val_y, val_base)
    val_corr_wape = wape(val_y, val_corrected)
    use_gpu_correction = bool(val_corr_wape <= val_base_wape)

    test_selected = test_corrected if use_gpu_correction else test_base

    metrics = pd.DataFrame(
        [
            {
                "split": "validation",
                "model": "baseline_rolling_mean",
                "wape": val_base_wape,
                "mae": mae(val_y, val_base),
                "bias": bias(val_y, val_base),
            },
            {
                "split": "validation",
                "model": "gpu_neural_residual_correction",
                "wape": val_corr_wape,
                "mae": mae(val_y, val_corrected),
                "bias": bias(val_y, val_corrected),
            },
            {
                "split": "test",
                "model": "baseline_rolling_mean",
                "wape": wape(test_y, test_base),
                "mae": mae(test_y, test_base),
                "bias": bias(test_y, test_base),
            },
            {
                "split": "test",
                "model": "gpu_neural_residual_correction",
                "wape": wape(test_y, test_corrected),
                "mae": mae(test_y, test_corrected),
                "bias": bias(test_y, test_corrected),
            },
            {
                "split": "test",
                "model": "validation_gated_selected_forecast",
                "wape": wape(test_y, test_selected),
                "mae": mae(test_y, test_selected),
                "bias": bias(test_y, test_selected),
            },
        ]
    )

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
    if len(bias_by_group):
        threshold = bias_by_group["abs_mean_bias"].quantile(0.8)
        high_bias_groups = bias_by_group[bias_by_group["abs_mean_bias"] >= threshold][group_cols]
        high_bias_key = set(map(tuple, high_bias_groups.to_numpy()))
        test_keys = list(map(tuple, test[group_cols].to_numpy()))
        high_bias_mask = np.array([k in high_bias_key for k in test_keys])
    else:
        high_bias_mask = np.ones(len(test), dtype=bool)

    if high_bias_mask.sum() > 0:
        hb_y = test_y[high_bias_mask]
        hb_base = test_base[high_bias_mask]
        hb_corr = test_corrected[high_bias_mask]
        high_bias_metrics = pd.DataFrame(
            [
                {
                    "slice": "high_bias_groups",
                    "model": "baseline_rolling_mean",
                    "rows": int(high_bias_mask.sum()),
                    "wape": wape(hb_y, hb_base),
                    "mae": mae(hb_y, hb_base),
                    "bias": bias(hb_y, hb_base),
                },
                {
                    "slice": "high_bias_groups",
                    "model": "gpu_neural_residual_correction",
                    "rows": int(high_bias_mask.sum()),
                    "wape": wape(hb_y, hb_corr),
                    "mae": mae(hb_y, hb_corr),
                    "bias": bias(hb_y, hb_corr),
                },
            ]
        )
    else:
        high_bias_metrics = pd.DataFrame()

    metrics_path = OUT_DIR / "gpu_residual_metrics.csv"
    high_bias_path = OUT_DIR / "gpu_residual_high_bias_slice_metrics.csv"
    report_path = REPORT_DIR / "gpu_residual_correction_report.md"

    metrics.to_csv(metrics_path, index=False)
    high_bias_metrics.to_csv(high_bias_path, index=False)

    test_base_wape = float(metrics[(metrics["split"] == "test") & (metrics["model"] == "baseline_rolling_mean")]["wape"].iloc[0])
    test_corr_wape = float(metrics[(metrics["split"] == "test") & (metrics["model"] == "gpu_neural_residual_correction")]["wape"].iloc[0])
    test_selected_wape = float(metrics[(metrics["split"] == "test") & (metrics["model"] == "validation_gated_selected_forecast")]["wape"].iloc[0])

    lines = [
        "# GPU Neural Residual Correction Report",
        "",
        "## GPU status",
        f"- Device: {meta['device']}",
        f"- Train rows: {meta['n_train']}",
        f"- Validation rows: {meta['n_val']}",
        f"- Test rows: {meta['n_test']}",
        f"- Features: {meta['n_features']}",
        f"- Training seconds: {meta['train_seconds']:.2f}",
        "",
        "## Validation gate",
        f"- Validation baseline WAPE: {val_base_wape:.6f}",
        f"- Validation GPU residual WAPE: {val_corr_wape:.6f}",
        f"- Use GPU correction on test: {use_gpu_correction}",
        "",
        "## Test comparison",
        f"- Test baseline WAPE: {test_base_wape:.6f}",
        f"- Test GPU residual WAPE: {test_corr_wape:.6f}",
        f"- Test validation-gated selected WAPE: {test_selected_wape:.6f}",
        "",
        "## Metrics table",
        metrics.to_markdown(index=False),
        "",
        "## High-bias slice metrics",
        high_bias_metrics.to_markdown(index=False) if len(high_bias_metrics) else "No high-bias slice rows available.",
        "",
        "## Claim boundary",
        "This is an optional GPU experiment using public/sample retail panels and validation-gated residual correction.",
        "It should not be described as a production deployment or as an Amazon internal model.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {metrics_path}")
    print(f"Wrote: {high_bias_path}")
    print(f"Wrote: {report_path}")
    print(f"GPU correction selected by validation gate: {use_gpu_correction}")


if __name__ == "__main__":
    main()
