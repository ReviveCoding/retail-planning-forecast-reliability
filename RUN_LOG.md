# Run Log

## Portable sample-fallback validation

Environment:
- Windows PowerShell
- Python 3.11.9
- Virtual environment: .venv

Commands:
python scripts\clean_outputs.py
python -m src.main --project all
python -m pytest -q

Result:
18 passed in 8.07s

Notes:
- configs/data_config.yml uses portable sample-fallback paths.
- Local real-data paths are stored separately in configs/data_config.local.yml and ignored by Git.
- tabulate is included because pandas DataFrame.to_markdown() requires it.
- Ridge baseline supports categorical features through one-hot encoding.
