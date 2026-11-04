import pandas as pd

from src.data.validate_schema import validate_weekly_schema
from src.utils import project_path


def test_weekly_schema_valid():
    df = pd.read_csv(project_path("outputs/processed/m5_weekly_panel.csv"))
    result = validate_weekly_schema(df)
    assert result["valid"], result
