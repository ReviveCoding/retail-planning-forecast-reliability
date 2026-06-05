from __future__ import annotations
import pandas as pd
from src.utils import load_yaml, write_csv, project_path
from src.data.ingest_m5 import ensure_m5_available, load_m5_long
from src.data.build_weekly_panel import build_weekly_panel
from src.data.validate_schema import validate_weekly_schema
from src.features.lag_rolling_features import make_supervised
from src.features.calendar_price_features import add_calendar_price_features
from src.forecasting.seasonal_naive import forecast_seasonal_naive
from src.forecasting.moving_average import forecast_moving_average
from src.forecasting.tree_baseline import train_tree_forecast
from src.forecasting.evaluate_forecasts import summarize_model_metrics
from src.planning.generate_synthetic_plan import generate_synthetic_plan
from src.planning.inventory_capacity_proxy import add_inventory_capacity_impacts
from src.planning.what_if_intervention import simulate_interventions, build_intervention_review_queue
from src.variance_bridge.plan_vs_actual import compute_plan_vs_actual_bridge
from src.variance_bridge.actual_vs_actual import compute_actual_vs_actual_bridge
from src.reporting.forecast_metrics_report import write_forecast_metrics_report
from src.reporting.variance_bridge_report import write_variance_bridge_report


def run_project1_core() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = load_yaml('configs/data_config.yml')
    forecast_cfg = load_yaml('configs/forecast_config.yml')
    plan_cfg = load_yaml('configs/plan_generation_config.yml')
    raw_dir = ensure_m5_available(project_path(data_cfg['paths']['m5_raw_dir']), data_cfg['sample'])
    long_df = load_m5_long(raw_dir, data_cfg.get('real_data_subset'))
    weekly = build_weekly_panel(long_df)
    schema = validate_weekly_schema(weekly)
    if not schema['valid']:
        raise ValueError(f'Weekly schema validation failed: {schema}')
    write_csv(weekly, project_path('outputs/processed/m5_weekly_panel.csv'))

    all_forecasts = []
    for horizon in forecast_cfg['horizons_weeks']:
        supervised = make_supervised(weekly, horizon)
        supervised = add_calendar_price_features(supervised)
        max_week = supervised['week_id'].max()
        test = supervised[supervised['week_id'] > max_week - forecast_cfg['test_weeks']].copy()
        all_forecasts.append(forecast_seasonal_naive(test, horizon))
        all_forecasts.append(forecast_moving_average(test, horizon))
        all_forecasts.append(train_tree_forecast(supervised, test_weeks=forecast_cfg['test_weeks'], random_state=forecast_cfg['models']['random_state']))
    forecast_df = pd.concat(all_forecasts, ignore_index=True)
    metrics = summarize_model_metrics(forecast_df)
    write_csv(forecast_df, project_path('outputs/forecasts/all_forecasts.csv'))
    write_csv(metrics, project_path('outputs/forecasts/forecast_metrics.csv'))

    best_row = metrics.sort_values('wape').iloc[0]
    best_forecast = forecast_df[(forecast_df['model'] == best_row['model']) & (forecast_df['forecast_horizon'] == best_row['forecast_horizon'])].copy()
    plan = generate_synthetic_plan(best_forecast, plan_cfg)
    plan = add_inventory_capacity_impacts(plan)
    write_csv(plan, project_path('outputs/plans/synthetic_plan.csv'))
    plan_bridge = compute_plan_vs_actual_bridge(plan)
    actual_bridge = compute_actual_vs_actual_bridge(plan)
    write_csv(plan_bridge, project_path('outputs/variance_bridge/plan_vs_actual_bridge.csv'))
    write_csv(actual_bridge, project_path('outputs/variance_bridge/actual_vs_actual_bridge.csv'))
    intervention = simulate_interventions(plan)
    review_keys = ['item_id','store_id','week_id','target_week_id','forecast_horizon']
    review_extra = [c for c in ['plan_capacity','plan_price','overstock_proxy_units','lost_sales_proxy_units','actual_units'] if c in plan.columns]
    queue_source = plan_bridge.merge(plan[review_keys + review_extra].drop_duplicates(review_keys), on=review_keys, how='left', suffixes=('', '_plan'))
    review_queue = build_intervention_review_queue(queue_source)
    write_csv(intervention, project_path('outputs/plans/what_if_interventions.csv'))
    write_csv(review_queue, project_path('outputs/plans/intervention_review_queue.csv'))
    write_forecast_metrics_report(metrics)
    write_variance_bridge_report(plan_bridge, actual_bridge)
    return weekly, forecast_df, metrics, plan
