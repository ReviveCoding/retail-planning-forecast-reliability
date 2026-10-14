from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict:
    """Load an absolute YAML path or resolve a relative path from the project root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dirs(*dirs: str | Path) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_md(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def project_path(*parts: str) -> Path:
    return ROOT.joinpath(*parts)
