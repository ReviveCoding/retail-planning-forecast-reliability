from __future__ import annotations
import pandas as pd
from src.utils import load_yaml, write_csv, project_path
from src.features.lag_rolling_features import make_supervised
from src.features.calendar_price_features import add_calendar_price_features
from src.forecasting.tree_baseline import train_tree_forecast
from src.forecasting.evaluate_forecasts import regression_metrics
from src.specialty.demand_state_timestep import add_demand_state_timestep
from src.specialty.structural_expected_demand import fit_structural_expected_demand
from src.specialty.residual_bias_correction import residual_corrected_forecast
from src.specialty.asymmetric_coverage_metrics import compute_asymmetric_coverage_metrics, summarize_coverage_metrics
from src.reporting.specialty_ablation_report import write_specialty_ablation_report


def run_project3_specialty(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_cfg = load_yaml('configs/forecast_config.yml')
    enhanced = add_demand_state_timestep(weekly, forecast_cfg.get('specialty', {}))
    horizon = forecast_cfg['horizons_weeks'][0]
    supervised = make_supervised(enhanced, horizon)
    supervised = add_calendar_price_features(supervised)

    tree_dst = train_tree_forecast(supervised, test_weeks=forecast_cfg['test_weeks'], random_state=forecast_cfg['models']['random_state']).copy()
    tree_dst['model'] = 'tabular_baseline_with_demand_state_timestep'

    model, structural_test = fit_structural_expected_demand(supervised, test_weeks=forecast_cfg['test_weeks'])
    structural_test['units_forecast'] = structural_test['expected_demand_path']
    structural_test['model'] = 'structural_expected_demand'

    raw_residual = residual_corrected_forecast(supervised, model, test_weeks=forecast_cfg['test_weeks'], gated=False, cfg=forecast_cfg.get('specialty', {}))
    gated_residual = residual_corrected_forecast(supervised, model, test_weeks=forecast_cfg['test_weeks'], gated=forecast_cfg.get('specialty', {}).get('gated_residual_correction', True), cfg=forecast_cfg.get('specialty', {}))

    specialty_forecasts = pd.concat([tree_dst, structural_test, raw_residual, gated_residual], ignore_index=True, sort=False)
    rows = []
    for model_name, g in specialty_forecasts.groupby('model'):
        m = regression_metrics(g['target_units'], g['units_forecast'])
        m.update({'model': model_name, 'forecast_horizon': horizon, 'n': len(g)})
        if 'selected_forecast_source' in g.columns:
            modes = g['selected_forecast_source'].dropna().unique().tolist()
            m['selected_forecast_source'] = ';'.join(map(str, modes)) if modes else ''
        if 'gate_dataset' in g.columns:
            m['gate_dataset'] = ';'.join(map(str, g['gate_dataset'].dropna().unique().tolist()))
        if 'validation_baseline_wape_for_gate' in g.columns:
            base = g['validation_baseline_wape_for_gate'].dropna()
            raw = g['validation_raw_wape_for_gate'].dropna()
            m['validation_baseline_wape_for_gate'] = float(base.iloc[0]) if len(base) else None
            m['validation_raw_wape_for_gate'] = float(raw.iloc[0]) if len(raw) else None
        rows.append(m)
    metrics = pd.DataFrame(rows)
    corrected = compute_asymmetric_coverage_metrics(gated_residual)
    coverage_summary = summarize_coverage_metrics(corrected)
    write_csv(enhanced, project_path('outputs/specialty/m5_weekly_with_demand_state_timestep.csv'))
    write_csv(specialty_forecasts, project_path('outputs/specialty/specialty_forecasts.csv'))
    write_csv(metrics, project_path('outputs/specialty/specialty_ablation_metrics.csv'))
    write_csv(corrected, project_path('outputs/specialty/asymmetric_coverage_detail.csv'))
    write_csv(coverage_summary, project_path('outputs/specialty/asymmetric_coverage_summary.csv'))
    write_specialty_ablation_report(metrics, coverage_summary)
    return specialty_forecasts, metrics, coverage_summary
