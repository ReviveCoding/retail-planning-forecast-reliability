from __future__ import annotations
import argparse
import pandas as pd
from src.utils import project_path
from src.reporting.claim_boundary_report import write_claim_boundary_report
from src.pipelines.common import setup_dirs
from src.pipelines.run_all import run_all
from src.pipelines.project1_core import run_project1_core
from src.pipelines.project2_driver_validation import run_project2_driver_validation
from src.pipelines.project3_specialty import run_project3_specialty
from src.pipelines.project4_genai_eval import run_project4_genai_eval


def _load_or_build_weekly():
    weekly_path = project_path('outputs/processed/m5_weekly_panel.csv')
    if weekly_path.exists():
        return pd.read_csv(weekly_path)
    return run_project1_core()[0]


def main():
    parser = argparse.ArgumentParser(description='Retail LTPF-style forecast intelligence framework')
    parser.add_argument('--project', default='all', choices=['all','sample','project1','project2','project3','project4'])
    args = parser.parse_args()
    setup_dirs()
    if args.project in ['all', 'sample']:
        run_all()
    elif args.project == 'project1':
        run_project1_core(); write_claim_boundary_report()
    elif args.project == 'project2':
        run_project2_driver_validation(); write_claim_boundary_report()
    elif args.project == 'project3':
        run_project3_specialty(_load_or_build_weekly()); write_claim_boundary_report()
    elif args.project == 'project4':
        if not project_path('outputs/variance_bridge/plan_vs_actual_bridge.csv').exists():
            weekly, forecast_df, forecast_metrics, plan = run_project1_core()
        else:
            forecast_metrics = pd.read_csv(project_path('outputs/forecasts/forecast_metrics.csv'))
        if project_path('outputs/driver_validation/driver_confidence_scores.csv').exists():
            driver_scores = pd.read_csv(project_path('outputs/driver_validation/driver_confidence_scores.csv'))
        else:
            driver_scores = run_project2_driver_validation()[0]
        plan_bridge = pd.read_csv(project_path('outputs/variance_bridge/plan_vs_actual_bridge.csv'))
        intervention = pd.read_csv(project_path('outputs/plans/what_if_interventions.csv'))
        run_project4_genai_eval(plan_bridge, driver_scores, intervention, forecast_metrics)
        write_claim_boundary_report()

if __name__ == '__main__':
    main()
