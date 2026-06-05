from __future__ import annotations
from pathlib import Path
import pandas as pd
from .generate_sample_m5_like import generate_sample_m5_like


def ensure_m5_available(raw_dir: str | Path, sample_cfg: dict) -> Path:
    raw_dir = Path(raw_dir)
    needed = ['sales_train_validation.csv', 'calendar.csv', 'sell_prices.csv']
    if not all((raw_dir / f).exists() for f in needed):
        generate_sample_m5_like(raw_dir, **sample_cfg)
    return raw_dir


def _apply_subset(sales: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame, subset_cfg: dict | None):
    """Apply a safe subset for large real M5 files.

    The generated sample data is already small, but the full M5 sales matrix is large.
    This option keeps local runs laptop-friendly and prevents accidental memory pressure.
    """
    subset_cfg = subset_cfg or {}
    if not subset_cfg.get('enabled', False):
        return sales, calendar, prices

    max_items = subset_cfg.get('max_items')
    max_stores = subset_cfg.get('max_stores')
    max_days = subset_cfg.get('max_days')

    if max_items:
        keep_items = sorted(sales['item_id'].dropna().unique())[: int(max_items)]
        sales = sales[sales['item_id'].isin(keep_items)].copy()
        prices = prices[prices['item_id'].isin(keep_items)].copy()
    if max_stores:
        keep_stores = sorted(sales['store_id'].dropna().unique())[: int(max_stores)]
        sales = sales[sales['store_id'].isin(keep_stores)].copy()
        prices = prices[prices['store_id'].isin(keep_stores)].copy()
    if max_days:
        d_cols = [c for c in sales.columns if c.startswith('d_')]
        keep_d = d_cols[-int(max_days):]
        id_cols = [c for c in sales.columns if not c.startswith('d_')]
        sales = sales[id_cols + keep_d].copy()
        calendar = calendar[calendar['d'].isin(keep_d)].copy()
    return sales, calendar, prices


def load_m5_long(raw_dir: str | Path, subset_cfg: dict | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / 'sales_train_validation.csv')
    calendar = pd.read_csv(raw_dir / 'calendar.csv', parse_dates=['date'])
    prices = pd.read_csv(raw_dir / 'sell_prices.csv')
    sales, calendar, prices = _apply_subset(sales, calendar, prices, subset_cfg)

    d_cols = [c for c in sales.columns if c.startswith('d_')]
    long_df = sales.melt(
        id_vars=['id','item_id','dept_id','cat_id','store_id','state_id'], value_vars=d_cols,
        var_name='d', value_name='units_actual'
    )
    long_df = long_df.merge(calendar[['d','date','wm_yr_wk','event_name_1','weekday','month']], on='d', how='left')
    long_df = long_df.merge(prices, on=['store_id','item_id','wm_yr_wk'], how='left')
    long_df['sell_price'] = long_df.groupby(['item_id','store_id'])['sell_price'].transform(lambda s: s.ffill().bfill())
    long_df['sell_price'] = long_df['sell_price'].fillna(long_df['sell_price'].median())
    long_df['event_flag'] = long_df['event_name_1'].fillna('').ne('').astype(int)
    long_df['holiday_flag'] = long_df['event_name_1'].fillna('').str.contains('Holiday', case=False).astype(int)
    long_df['promotion_flag'] = 0
    return long_df
