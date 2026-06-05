from __future__ import annotations
import pandas as pd


def build_weekly_panel(long_df: pd.DataFrame) -> pd.DataFrame:
    df = long_df.copy()
    df['week_id'] = df['wm_yr_wk'].astype(int)
    weekly = df.groupby(['item_id','store_id','cat_id','dept_id','state_id','week_id'], as_index=False).agg(
        units_actual=('units_actual','sum'),
        price_actual=('sell_price','mean'),
        event_flag=('event_flag','max'),
        holiday_flag=('holiday_flag','max'),
        promotion_flag=('promotion_flag','max')
    )
    weekly['category_id'] = weekly['cat_id']
    weekly['region_id'] = weekly['state_id']
    weekly['revenue_actual'] = weekly['units_actual'] * weekly['price_actual']
    weekly = weekly.sort_values(['item_id','store_id','week_id']).reset_index(drop=True)
    return weekly
