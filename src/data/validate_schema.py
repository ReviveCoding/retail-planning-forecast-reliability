from __future__ import annotations
import pandas as pd

REQUIRED_WEEKLY_COLUMNS = {'item_id','store_id','week_id','units_actual','price_actual','revenue_actual'}


def validate_weekly_schema(df: pd.DataFrame) -> dict:
    missing = sorted(REQUIRED_WEEKLY_COLUMNS - set(df.columns))
    negative_units = int((df.get('units_actual', pd.Series(dtype=float)) < 0).sum())
    negative_prices = int((df.get('price_actual', pd.Series(dtype=float)) <= 0).sum())
    duplicates = int(df.duplicated(['item_id','store_id','week_id']).sum()) if all(c in df for c in ['item_id','store_id','week_id']) else -1
    valid = not missing and negative_units == 0 and negative_prices == 0 and duplicates == 0
    return {'valid': valid, 'missing_columns': missing, 'negative_units': negative_units, 'nonpositive_prices': negative_prices, 'duplicate_item_store_weeks': duplicates}
