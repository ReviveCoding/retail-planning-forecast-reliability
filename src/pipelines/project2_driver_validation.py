from __future__ import annotations
import pandas as pd
from src.utils import load_yaml, write_csv, project_path
from src.data.ingest_favorita import ensure_favorita_available, load_favorita_weekly
from src.driver_validation.driver_confidence_score import score_all_drivers
from src.driver_validation.confidence_calibration import confidence_calibration
from src.reporting.driver_validation_report import write_driver_validation_report


def run_project2_driver_validation() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = load_yaml('configs/data_config.yml')
    confidence_cfg = load_yaml('configs/confidence_config.yml')
    driver_cfg = load_yaml('configs/driver_registry.yml')
    raw_dir = ensure_favorita_available(project_path(data_cfg['paths']['favorita_raw_dir']))
    fav = load_favorita_weekly(raw_dir)
    subset = data_cfg.get('driver_validation_subset', {})
    if subset.get('enabled', True):
        keep_items = sorted(fav['item_id'].dropna().unique())[: int(subset.get('max_items', 30))]
        keep_stores = sorted(fav['store_id'].dropna().unique())[: int(subset.get('max_stores', 6))]
        fav = fav[fav['item_id'].isin(keep_items) & fav['store_id'].isin(keep_stores)].copy()
        max_weeks = int(subset.get('max_weeks', 80))
        if max_weeks > 0:
            keep_weeks = sorted(fav['week_id'].dropna().unique())[-max_weeks:]
            fav = fav[fav['week_id'].isin(keep_weeks)].copy()
    write_csv(fav, project_path('outputs/processed/favorita_weekly_panel.csv'))
    drivers = [d['treatment_col'] for d in driver_cfg['drivers']]
    weights = confidence_cfg['confidence_weights']
    scores = score_all_drivers(
        fav, drivers, weights=weights,
        placebo_iterations=confidence_cfg.get('placebo_iterations', 200),
        bootstrap_samples=confidence_cfg.get('bootstrap_samples', 200)
    )
    calib = confidence_calibration(scores)
    write_csv(scores, project_path('outputs/driver_validation/driver_confidence_scores.csv'))
    write_csv(calib, project_path('outputs/confidence/confidence_calibration.csv'))
    write_driver_validation_report(scores, calib)
    return scores, calib
