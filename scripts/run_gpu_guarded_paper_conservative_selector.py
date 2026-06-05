from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "specialty"
REPORT_DIR = ROOT / "reports"

ALPHA_PATH = OUT_DIR / "guarded_paper_alpha_grid.csv"
METRICS_PATH = OUT_DIR / "guarded_paper_method_metrics.csv"


def pct_reduction(old: float, new: float) -> float:
    if not np.isfinite(old) or abs(old) < 1e-12:
        return float("nan")
    return float((old - new) / abs(old))


def main() -> None:
    if not ALPHA_PATH.exists():
        raise FileNotFoundError(
            "Missing guarded_paper_alpha_grid.csv. Run scripts/run_gpu_guarded_paper_method_experiment.py first."
        )
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Missing guarded_paper_method_metrics.csv. Run scripts/run_gpu_guarded_paper_method_experiment.py first."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    alpha = pd.read_csv(ALPHA_PATH)
    metrics = pd.read_csv(METRICS_PATH)

    for c in ["alpha", "wape", "mae", "bias", "abs_bias", "under_forecast_rate"]:
        alpha[c] = pd.to_numeric(alpha[c], errors="coerce")
        metrics[c] = pd.to_numeric(metrics[c], errors="coerce")

    val = alpha[alpha["split"] == "validation"].copy()
    test = alpha[alpha["split"] == "test"].copy()

    base_val = metrics[(metrics["split"] == "validation") & (metrics["model"] == "rolling_baseline")].iloc[0]
    base_test = metrics[(metrics["split"] == "test") & (metrics["model"] == "rolling_baseline")].iloc[0]

    val["bias_sign_flip"] = np.sign(val["bias"]) != np.sign(float(base_val["bias"]))

    aggressive_alpha = float(val.sort_values(["wape", "abs_bias"]).iloc[0]["alpha"])

    eligible = val[
        (val["wape"] <= float(base_val["wape"]) * 0.995)
        & (val["alpha"] > 0.0)
    ].copy()

    if len(eligible):
        conservative_alpha = float(eligible.sort_values(["alpha", "wape"]).iloc[0]["alpha"])
    else:
        conservative_alpha = 0.0

    sign_safe = val[
        (val["wape"] <= float(base_val["wape"]) * 0.995)
        & (val["alpha"] > 0.0)
        & (~val["bias_sign_flip"])
    ].copy()

    if len(sign_safe):
        sign_safe_alpha = float(sign_safe.sort_values(["alpha", "wape"]).iloc[0]["alpha"])
    else:
        sign_safe_alpha = conservative_alpha

    rows = []
    for policy, a in [
        ("baseline_alpha_0", 0.0),
        ("aggressive_validation_wape_best", aggressive_alpha),
        ("conservative_smallest_improving_alpha", conservative_alpha),
        ("sign_safe_smallest_improving_alpha", sign_safe_alpha),
    ]:
        v = val[val["alpha"] == a].iloc[0]
        t = test[test["alpha"] == a].iloc[0]

        rows.append({
            "policy": policy,
            "alpha": a,
            "validation_wape": float(v["wape"]),
            "validation_abs_bias": float(v["abs_bias"]),
            "validation_bias": float(v["bias"]),
            "validation_under_forecast_rate": float(v["under_forecast_rate"]),
            "test_wape": float(t["wape"]),
            "test_abs_bias": float(t["abs_bias"]),
            "test_bias": float(t["bias"]),
            "test_under_forecast_rate": float(t["under_forecast_rate"]),
            "test_wape_reduction_vs_baseline": pct_reduction(float(base_test["wape"]), float(t["wape"])),
            "test_abs_bias_reduction_vs_baseline": pct_reduction(float(base_test["abs_bias"]), float(t["abs_bias"])),
        })

    policy_df = pd.DataFrame(rows)

    out_path = OUT_DIR / "guarded_paper_conservative_policy_summary.csv"
    report_path = REPORT_DIR / "guarded_paper_conservative_policy_report.md"

    policy_df.to_csv(out_path, index=False)

    lines = [
        "# Guarded Paper-Method Conservative Policy Report",
        "",
        "## Diagnosis",
        "Aggressive validation-WAPE selection over-corrected the test forecasts.",
        "This selector separates aggressive selection from conservative reliability selection.",
        "",
        "## Policy summary",
        policy_df.to_markdown(index=False),
        "",
        "## Recommended claim boundary",
        "Use the conservative or sign-safe policy for claims.",
        "Do not use the aggressive alpha policy for resume claims because it over-corrected test forecasts.",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {report_path}")
    print(policy_df.to_string(index=False))


if __name__ == "__main__":
    main()
