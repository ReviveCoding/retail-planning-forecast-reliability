# Retail LTPF-Style Forecast Intelligence, Driver Validation, and Planning Narrative Framework

A local/offline end-to-end framework that connects public or sample retail demand data to forecast baselines, synthetic planning assumptions, Plan-vs-Actual and Actual-vs-Actual variance bridges, public-data driver validation, specialty-enhanced risk-adjusted forecasting, and evidence-grounded planning narratives.

This repository was designed as an application-ready portfolio project for long-term planning and forecasting roles. It emphasizes reproducible forecasting workflows, variance decomposition, driver validation, confidence scoring, decision-support reporting, and careful claim boundaries.

## What this repo demonstrates

1. **Core forecasting and variance bridge**: M5-style weekly item-store demand forecasting, synthetic plan generation, and accounting-identity variance decomposition.
2. **Driver validation and confidence scoring**: Favorita-style event-study and DiD-style driver validation with placebo tests, bootstrap intervals, pre-trend diagnostics, and confidence scores.
3. **Specialty-enhanced forecasting and risk planning**: leakage-safe demand-state timestep features, structural expected demand, gated residual correction, and asymmetric coverage-vs-overreach planning metrics.
4. **GenAI narrative evaluation**: deterministic or optional LLM-ready planning narratives evaluated for numeric consistency, driver faithfulness, caveat coverage, unsupported-claim risk, and corrupted-narrative robustness.

## Claim boundary

This project uses public retail data if available or generated sample retail data. Planning, inventory, capacity, and intervention fields are synthetic/proxy assumptions. This is a local/offline analytical framework. It is not an Amazon internal system, does not use proprietary Amazon data, and is not a production deployment.


## Optimized engineering structure

The repository separates source code from generated artifacts. `outputs/` and `reports/` are created by `make run-all` and are intentionally ignored by Git. Pipeline orchestration lives under `src/pipelines/`, while `src/main.py` is a thin CLI wrapper. Tests are split into fast unit tests and end-to-end integration tests so local development does not require rerunning the full pipeline after every small change.

Useful commands:

```bash
make run-core          # Project 1 only
make run-driver        # Project 2 only
make run-specialty     # Project 3 only
make run-narrative     # Project 4 only
make run-all           # full integrated pipeline
make test-unit         # fast tests
make test-integration  # full pipeline validation
make clean             # cross-platform cleanup
```

## Quick start

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
make run-all
make test
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.main --project all
python -m pytest -q

# Faster development checks
python -m pytest -q tests/unit
python -m pytest -q tests/integration
```

If real M5 files are not placed under `data/raw/m5/`, the pipeline automatically generates sample M5-like data. If real Favorita files are not placed under `data/raw/favorita/`, it generates sample Favorita-like data.

## Real public-data guardrails

Full public M5 and Favorita files can be large. The M5 adapter includes a local subset guard in `configs/data_config.yml`:

```yaml
real_data_subset:
  enabled: true
  max_items: 250
  max_stores: 5
  max_days: 730
```

Keep this enabled for the first real-data run. See `docs/REAL_DATA_GUIDE.md` for file placement and caveats.

## Expected outputs

- `outputs/forecasts/forecast_metrics.csv`
- `outputs/variance_bridge/plan_vs_actual_bridge.csv`
- `outputs/variance_bridge/actual_vs_actual_bridge.csv`
- `outputs/driver_validation/driver_confidence_scores.csv`
- `outputs/specialty/specialty_ablation_metrics.csv`
- `outputs/genai_eval/narrative_eval_metrics.csv`
- `reports/executive_ltp_review.md`
- `reports/forecast_metrics_report.md`
- `reports/variance_bridge_report.md`
- `reports/driver_validation_report.md`
- `reports/specialty_ablation_report.md`
- `reports/genai_narrative_eval_report.md`
- `reports/claim_boundary.md`

## Repo structure

```text
configs/              YAML configs for data, forecasts, plan generation, confidence, narrative evaluation, and claim boundary
src/data/             sample data generation, M5/Favorita ingestion, weekly panel building, validation, leakage checks
src/forecasting/      seasonal naive, moving average, tabular baseline, metrics
src/planning/         synthetic plan and proxy inventory/capacity generation
src/variance_bridge/  Plan-vs-Actual and Actual-vs-Actual decomposition with identity checks
src/driver_validation/ event study, DiD-style validation, placebo tests, bootstrap CI, confidence scoring
src/specialty/        demand-state timestep, residual correction, asymmetric coverage metrics
src/genai_eval/       template narrative and evidence-grounded evaluation checks
src/reporting/        markdown report generation
tests/unit/           fast function-level tests that do not generate artifacts
tests/integration/    end-to-end tests that run the full local pipeline once and validate generated outputs
docs/                 methodology, roadmap, real-data guide, Windows run guide, and resume-safe claim guidance
```

## Four integrated subprojects

### Project 1: Core Forecasting & Variance Bridge
Creates M5-style weekly item-store panels, forecasts 1-week and 4-week demand, generates synthetic plans, and decomposes forecast-plan-actual gaps into unit, price, interaction, inventory proxy, capacity proxy, and residual effects. It also creates an intervention review queue so aggregate variance results map back to item-store actions.

### Project 2: Driver Validation & Confidence Scoring
Uses Favorita-style promotion, holiday, transaction, and oil covariates to validate demand-driver hypotheses through event-study, DiD-style comparison, placebo tests, bootstrap intervals, pre-trend diagnostics, and confidence calibration.

### Project 3: Specialty-Enhanced Forecasting & Risk-Adjusted Planning
Adds leakage-safe demand-state timestep features, mechanism-aware expected demand, validation-gated residual correction, and asymmetric demand-coverage metrics inspired by reliability and localization research.

### Project 4: GenAI Planning Narrative Evaluation
Turns quantitative tables into stakeholder-readable summaries and evaluates whether the narrative stays faithful to source metrics and caveats. The module also generates controlled corrupted narratives to test whether the evaluation layer catches numeric hallucination, driver swapping, missing caveats, and unsupported claims.
