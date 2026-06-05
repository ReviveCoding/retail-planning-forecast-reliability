from __future__ import annotations
from src.utils import ensure_dirs, project_path

OUTPUT_DIRS = [
    'outputs/processed','outputs/forecasts','outputs/plans','outputs/variance_bridge','outputs/metrics',
    'outputs/driver_validation','outputs/confidence','outputs/specialty','outputs/genai_eval','reports'
]


def setup_dirs() -> None:
    ensure_dirs(*(project_path(d) for d in OUTPUT_DIRS))
