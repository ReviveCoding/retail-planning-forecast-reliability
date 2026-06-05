from __future__ import annotations
import pandas as pd
from src.forecasting.tree_baseline import FEATURE_COLS
from src.forecasting.simple_ridge import ridge_fit_predict
from src.forecasting.evaluate_forecasts import regression_metrics


def _wape(df: pd.DataFrame, forecast_col='units_forecast') -> float:
    return regression_metrics(df['target_units'], df[forecast_col])['wape']


def _tree_fit_predict(train: pd.DataFrame, predict_df: pd.DataFrame, random_state: int = 42) -> pd.Series:
    feature_cols = [c for c in FEATURE_COLS if c in train.columns]
    pred = ridge_fit_predict(train, predict_df, feature_cols, target_col='target_units', alpha=1.0)
    return pd.Series(pred, index=predict_df.index).clip(lower=0)


def _residual_fit_predict(train: pd.DataFrame, predict_df: pd.DataFrame, structural_model, feature_cols: list[str], random_state: int = 42) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    structural_features = [c for c in ['week_sin','week_cos','price_actual','event_flag','holiday_flag','promotion_flag','category_id','region_id'] if c in train.columns]
    train = train.copy()
    predict_df = predict_df.copy()
    train['expected_demand_path'] = structural_model.predict(train[structural_features]).clip(lower=0)
    predict_df['expected_demand_path'] = structural_model.predict(predict_df[structural_features]).clip(lower=0)
    train['residual_target'] = train['target_units'] - train['expected_demand_path']
    cols = [c for c in feature_cols + ['expected_demand_path'] if c in train.columns]
    residual_prediction = ridge_fit_predict(train, predict_df, cols, target_col='residual_target', alpha=1.0)
    forecast = (predict_df['expected_demand_path'] + residual_prediction).clip(lower=0)
    predict_df['residual_prediction'] = residual_prediction
    predict_df['raw_residual_corrected_forecast'] = forecast
    return pd.Series(forecast, index=predict_df.index), train, predict_df


def residual_corrected_forecast(supervised: pd.DataFrame, structural_model, test_weeks=12, gated: bool = True, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    validation_weeks = int(cfg.get('validation_weeks', max(4, test_weeks // 2)))
    validation_weeks = max(1, min(validation_weeks, max(1, test_weeks - 1)))
    random_state = int(cfg.get('random_state', 42))
    max_week = supervised['week_id'].max()
    test_start_week = max_week - test_weeks + 1
    val_start_week = test_start_week - validation_weeks
    train = supervised[supervised['week_id'] < val_start_week].copy()
    val = supervised[(supervised['week_id'] >= val_start_week) & (supervised['week_id'] < test_start_week)].copy()
    test = supervised[supervised['week_id'] >= test_start_week].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError('Not enough weekly observations for train/validation/test residual-correction split.')
    feature_cols = [c for c in FEATURE_COLS if c in supervised.columns]

    val_base = val.copy()
    val_base['units_forecast'] = _tree_fit_predict(train, val, random_state=random_state)
    val_raw_pred, _, _ = _residual_fit_predict(train, val, structural_model, feature_cols, random_state=random_state)
    val_raw = val.copy()
    val_raw['units_forecast'] = val_raw_pred
    validation_baseline_wape = _wape(val_base)
    validation_raw_wape = _wape(val_raw)
    use_residual = (validation_raw_wape <= validation_baseline_wape) if gated else True

    train_plus_val = pd.concat([train, val], ignore_index=True)
    if use_residual:
        test_pred, _, test_aug = _residual_fit_predict(train_plus_val, test, structural_model, feature_cols, random_state=random_state)
        out = test.copy()
        out['units_forecast'] = test_pred
        out['raw_residual_corrected_forecast'] = test_aug['raw_residual_corrected_forecast'].values
        out['residual_prediction'] = test_aug['residual_prediction'].values
        out['expected_demand_path'] = test_aug['expected_demand_path'].values
        out['selected_forecast_source'] = 'raw_residual_correction_selected_on_validation'
    else:
        out = test.copy()
        out['units_forecast'] = _tree_fit_predict(train_plus_val, test, random_state=random_state)
        out['selected_forecast_source'] = 'tabular_baseline_fallback_selected_on_validation'
    out['model'] = 'gated_business_mechanism_residual_correction' if gated else 'business_mechanism_residual_correction_raw'
    out['gate_dataset'] = 'validation'
    out['validation_baseline_wape_for_gate'] = validation_baseline_wape
    out['validation_raw_wape_for_gate'] = validation_raw_wape
    out['gate_selected_residual'] = bool(use_residual)
    return out
