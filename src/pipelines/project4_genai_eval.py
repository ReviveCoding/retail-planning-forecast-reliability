from __future__ import annotations
import pandas as pd
from src.utils import load_yaml, write_csv, project_path
from src.genai_eval.template_narrative import generate_template_narrative, generate_corrupted_narratives
from src.genai_eval.numeric_consistency import numeric_consistency
from src.genai_eval.unsupported_claim_detector import unsupported_claim_rate
from src.genai_eval.driver_faithfulness import driver_faithfulness
from src.genai_eval.caveat_coverage import caveat_coverage
from src.genai_eval.human_review_router import route_for_review
from src.reporting.genai_narrative_eval_report import write_genai_eval_report
from src.reporting.executive_report import write_executive_report


def run_project4_genai_eval(plan_bridge: pd.DataFrame, driver_scores: pd.DataFrame, intervention: pd.DataFrame, forecast_metrics: pd.DataFrame):
    narrative_cfg = load_yaml('configs/narrative_eval_config.yml')
    comp_cols = ['unit_effect','price_effect','interaction_effect','inventory_proxy_effect','capacity_proxy_effect','residual_effect']
    comp_sums = plan_bridge[comp_cols].sum(numeric_only=True)
    largest_component = comp_sums.abs().sort_values(ascending=False).index[0]
    variance_summary = {
        'total_revenue_gap': float(plan_bridge['total_revenue_gap'].sum()),
        'largest_component': largest_component,
        'largest_component_value': float(comp_sums[largest_component]),
    }
    top_driver = driver_scores.iloc[0].to_dict() if len(driver_scores) else {'driver':'not_available','driver_confidence':0.0}
    narrative = generate_template_narrative(variance_summary, top_driver)
    cases = generate_corrupted_narratives(narrative, variance_summary, top_driver)
    expected_numbers = [variance_summary['total_revenue_gap'], variance_summary['largest_component_value'], float(top_driver.get('driver_confidence', 0))]
    eval_rows = []
    for case in cases:
        case_narrative = case['narrative']
        eval_metrics = {
            'case_id': case['case_id'],
            'expected_issue': case['expected_issue'],
            'numeric_consistency': numeric_consistency(case_narrative, expected_numbers),
            'unsupported_claim_rate': unsupported_claim_rate(case_narrative),
            'driver_faithfulness': driver_faithfulness(case_narrative, top_driver.get('driver','not_available')),
            'caveat_coverage': caveat_coverage(case_narrative, narrative_cfg['required_caveats']),
        }
        eval_metrics['human_review_required'] = route_for_review(eval_metrics, narrative_cfg['review_thresholds'])
        eval_rows.append(eval_metrics)
    eval_df = pd.DataFrame(eval_rows)
    write_csv(pd.DataFrame(cases), project_path('outputs/genai_eval/generated_narratives.csv'))
    write_csv(eval_df, project_path('outputs/genai_eval/narrative_eval_metrics.csv'))
    write_genai_eval_report(eval_df, narrative)
    write_executive_report(forecast_metrics, plan_bridge, driver_scores, intervention, narrative)
    return narrative, eval_df
