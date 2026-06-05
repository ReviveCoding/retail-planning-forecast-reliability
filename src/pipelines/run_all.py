from __future__ import annotations
import pandas as pd
from src.utils import project_path
from src.reporting.claim_boundary_report import write_claim_boundary_report
from src.pipelines.common import setup_dirs
from src.pipelines.project1_core import run_project1_core
from src.pipelines.project2_driver_validation import run_project2_driver_validation
from src.pipelines.project3_specialty import run_project3_specialty
from src.pipelines.project4_genai_eval import run_project4_genai_eval


def run_all() -> bool:
    setup_dirs()
    weekly, forecast_df, forecast_metrics, plan = run_project1_core()
    driver_scores, calib = run_project2_driver_validation()
    run_project3_specialty(weekly)
    plan_bridge = pd.read_csv(project_path('outputs/variance_bridge/plan_vs_actual_bridge.csv'))
    intervention = pd.read_csv(project_path('outputs/plans/what_if_interventions.csv'))
    run_project4_genai_eval(plan_bridge, driver_scores, intervention, forecast_metrics)
    write_claim_boundary_report()
    return True
