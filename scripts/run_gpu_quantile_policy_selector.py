from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "specialty"
REPORT_DIR = ROOT / "reports"

METRICS_PATH = OUT_DIR / "gpu_quantile_asymmetric_buffer_metrics.csv"
HIGH_BIAS_PATH = OUT_DIR / "gpu_quantile_asymmetric_buffer_high_bias_metrics.csv"


def add_cost_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in [
        "coverage_recall",
        "buffer_precision",
        "coverage_f1",
        "lost_sales_proxy",
        "overstock_proxy",
        "under_forecast_rate",
        "wape",
        "mae",
        "bias",
        "abs_bias",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Balanced cost: treats lost sales and excess buffer as both costly.
    out["balanced_cost"] = (
        1.00 * out["lost_sales_proxy"]
        + 1.00 * out["overstock_proxy"]
        + 0.10 * out["under_forecast_rate"]
    )

    # Service policy: prioritizes avoiding under-forecast and lost sales.
    out["service_cost"] = (
        3.00 * out["lost_sales_proxy"]
        + 0.75 * out["overstock_proxy"]
        + 0.25 * out["under_forecast_rate"]
    )

    # Conservative policy: penalizes overstock and WAPE more heavily.
    out["conservative_cost"] = (
        1.00 * out["lost_sales_proxy"]
        + 2.00 * out["overstock_proxy"]
        + 0.50 * out["wape"]
    )

    return out


def select_policy(val_df: pd.DataFrame, policy_name: str) -> pd.Series:
    baseline = val_df[val_df["model"] == "baseline_point"].iloc[0]

    if policy_name == "global_coverage_f1_policy":
        # Maximize coverage F1 while rejecting extreme overstock.
        max_overstock = max(float(baseline["overstock_proxy"]) + 0.05, 0.10)
        eligible = val_df[val_df["overstock_proxy"] <= max_overstock].copy()
        if not len(eligible):
            eligible = val_df.copy()
        return eligible.sort_values(
            ["coverage_f1", "lost_sales_proxy", "buffer_precision"],
            ascending=[False, True, False],
        ).iloc[0]

    if policy_name == "balanced_cost_policy":
        return val_df.sort_values(["balanced_cost", "coverage_f1"], ascending=[True, False]).iloc[0]

    if policy_name == "service_level_policy":
        # For service-level decisions, accept more buffer if it sharply reduces lost sales.
        return val_df.sort_values(["service_cost", "lost_sales_proxy", "coverage_recall"], ascending=[True, True, False]).iloc[0]

    if policy_name == "conservative_cost_policy":
        return val_df.sort_values(["conservative_cost", "coverage_f1"], ascending=[True, False]).iloc[0]

    raise ValueError(policy_name)


def pct_change(old: float, new: float, higher_is_better: bool) -> float:
    if not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    raw = (new - old) / abs(old)
    return float(raw if higher_is_better else -raw)


def summarize_against_baseline(df: pd.DataFrame, selected_model: str, split: str) -> dict:
    base = df[(df["split"] == split) & (df["model"] == "baseline_point")].iloc[0]
    sel = df[(df["split"] == split) & (df["model"] == selected_model)].iloc[0]

    return {
        "split": split,
        "selected_model": selected_model,
        "baseline_coverage_f1": float(base["coverage_f1"]),
        "selected_coverage_f1": float(sel["coverage_f1"]),
        "coverage_f1_rel_change": pct_change(float(base["coverage_f1"]), float(sel["coverage_f1"]), True),
        "baseline_lost_sales_proxy": float(base["lost_sales_proxy"]),
        "selected_lost_sales_proxy": float(sel["lost_sales_proxy"]),
        "lost_sales_rel_reduction": pct_change(float(base["lost_sales_proxy"]), float(sel["lost_sales_proxy"]), False),
        "baseline_overstock_proxy": float(base["overstock_proxy"]),
        "selected_overstock_proxy": float(sel["overstock_proxy"]),
        "overstock_rel_reduction": pct_change(float(base["overstock_proxy"]), float(sel["overstock_proxy"]), False),
        "baseline_under_forecast_rate": float(base["under_forecast_rate"]),
        "selected_under_forecast_rate": float(sel["under_forecast_rate"]),
        "under_forecast_rel_reduction": pct_change(float(base["under_forecast_rate"]), float(sel["under_forecast_rate"]), False),
        "baseline_wape": float(base["wape"]),
        "selected_wape": float(sel["wape"]),
        "wape_rel_reduction": pct_change(float(base["wape"]), float(sel["wape"]), False),
    }


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"{METRICS_PATH} not found. Run: python scripts/run_gpu_quantile_asymmetric_buffer_experiment.py"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = add_cost_columns(pd.read_csv(METRICS_PATH))
    val = metrics[metrics["split"] == "validation"].copy()
    test = metrics[metrics["split"] == "test"].copy()

    policies = [
        "global_coverage_f1_policy",
        "balanced_cost_policy",
        "service_level_policy",
        "conservative_cost_policy",
    ]

    selection_rows = []
    comparison_rows = []

    for policy in policies:
        selected_val = select_policy(val, policy)
        model = str(selected_val["model"])
        selected_test = test[test["model"] == model].iloc[0]

        selection_rows.append(
            {
                "policy": policy,
                "selected_model": model,
                "validation_coverage_f1": float(selected_val["coverage_f1"]),
                "validation_lost_sales_proxy": float(selected_val["lost_sales_proxy"]),
                "validation_overstock_proxy": float(selected_val["overstock_proxy"]),
                "validation_balanced_cost": float(selected_val["balanced_cost"]),
                "test_coverage_f1": float(selected_test["coverage_f1"]),
                "test_lost_sales_proxy": float(selected_test["lost_sales_proxy"]),
                "test_overstock_proxy": float(selected_test["overstock_proxy"]),
                "test_under_forecast_rate": float(selected_test["under_forecast_rate"]),
                "test_wape": float(selected_test["wape"]),
            }
        )

        row = summarize_against_baseline(metrics, model, "test")
        row["policy"] = policy
        comparison_rows.append(row)

    selection_df = pd.DataFrame(selection_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    high_bias_summary = pd.DataFrame()
    if HIGH_BIAS_PATH.exists():
        hb = add_cost_columns(pd.read_csv(HIGH_BIAS_PATH))
        base_hb = hb[hb["model"] == "baseline_point"].iloc[0]

        hb_rows = []
        for _, row in hb.iterrows():
            hb_rows.append(
                {
                    "model": row["model"],
                    "policy_group": row["policy_group"],
                    "coverage_f1": float(row["coverage_f1"]),
                    "lost_sales_proxy": float(row["lost_sales_proxy"]),
                    "overstock_proxy": float(row["overstock_proxy"]),
                    "under_forecast_rate": float(row["under_forecast_rate"]),
                    "abs_bias": float(row["abs_bias"]),
                    "coverage_f1_rel_change_vs_baseline": pct_change(float(base_hb["coverage_f1"]), float(row["coverage_f1"]), True),
                    "lost_sales_rel_reduction_vs_baseline": pct_change(float(base_hb["lost_sales_proxy"]), float(row["lost_sales_proxy"]), False),
                    "abs_bias_rel_reduction_vs_baseline": pct_change(float(base_hb["abs_bias"]), float(row["abs_bias"]), False),
                }
            )

        high_bias_summary = pd.DataFrame(hb_rows).sort_values(
            ["abs_bias_rel_reduction_vs_baseline", "coverage_f1_rel_change_vs_baseline"],
            ascending=[False, False],
        )

    selection_path = OUT_DIR / "gpu_quantile_policy_selection_summary.csv"
    comparison_path = OUT_DIR / "gpu_quantile_policy_vs_baseline_summary.csv"
    high_bias_summary_path = OUT_DIR / "gpu_quantile_high_bias_diagnostic_summary.csv"
    report_path = REPORT_DIR / "gpu_quantile_policy_selector_report.md"

    selection_df.to_csv(selection_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    high_bias_summary.to_csv(high_bias_summary_path, index=False)

    best_hb_line = "No high-bias diagnostic rows available."
    if len(high_bias_summary):
        best_hb = high_bias_summary.iloc[0]
        best_hb_line = (
            f"- Best high-bias diagnostic model: {best_hb['model']}\n"
            f"- High-bias abs-bias reduction vs baseline: {best_hb['abs_bias_rel_reduction_vs_baseline']:.2%}\n"
            f"- High-bias lost-sales reduction vs baseline: {best_hb['lost_sales_rel_reduction_vs_baseline']:.2%}\n"
            f"- High-bias coverage-F1 change vs baseline: {best_hb['coverage_f1_rel_change_vs_baseline']:.2%}"
        )

    lines = [
        "# GPU Quantile Cost-Sensitive Policy Selector Report",
        "",
        "## Purpose",
        "The quantile model should not be judged only by point-forecast WAPE.",
        "This report adds validation-selected policy families for planning tradeoffs: coverage F1, balanced cost, service-level risk, and conservative cost.",
        "",
        "## Validation-selected policies",
        selection_df.to_markdown(index=False),
        "",
        "## Test comparison versus baseline point forecast",
        comparison_df.to_markdown(index=False),
        "",
        "## High-bias slice diagnostic",
        "High-bias rows are reported as diagnostics, not used for global policy selection.",
        best_hb_line,
        "",
        high_bias_summary.to_markdown(index=False) if len(high_bias_summary) else "No high-bias diagnostic rows available.",
        "",
        "## Claim boundary",
        "Policy selection uses validation metrics only.",
        "Test metrics and high-bias diagnostic rows are for reporting and claim calibration.",
        "Do not claim the GPU quantile model is the global winner unless the selected validation policy also wins on the relevant test metric.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {selection_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {high_bias_summary_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
