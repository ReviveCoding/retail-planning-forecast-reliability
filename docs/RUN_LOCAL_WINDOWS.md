# Local Run Guide for Windows PowerShell

Use this guide if `make` is not available on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.main --project all
python -m pytest -q
```

Expected core outputs:

```text
outputs/forecasts/forecast_metrics.csv
outputs/variance_bridge/plan_vs_actual_bridge.csv
outputs/variance_bridge/actual_vs_actual_bridge.csv
outputs/driver_validation/driver_confidence_scores.csv
outputs/specialty/specialty_ablation_metrics.csv
outputs/genai_eval/narrative_eval_metrics.csv
reports/executive_ltp_review.md
```

If you want to reset generated artifacts on Windows:

```powershell
Remove-Item -Recurse -Force outputs\*, reports\*.md -ErrorAction SilentlyContinue
```
