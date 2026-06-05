from __future__ import annotations
from pathlib import Path
import pandas as pd
from .generate_sample_favorita_like import generate_sample_favorita_like


def ensure_favorita_available(raw_dir: str | Path) -> Path:
    raw_dir = Path(raw_dir)
    needed = ['train.csv', 'transactions.csv', 'holidays_events.csv', 'oil.csv', 'items.csv', 'stores.csv']
    if not all((raw_dir / f).exists() for f in needed):
        generate_sample_favorita_like(raw_dir)
    return raw_dir


def load_favorita_weekly(raw_dir: str | Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / 'train.csv', parse_dates=['date'])
    tx = pd.read_csv(raw_dir / 'transactions.csv', parse_dates=['date'])
    holidays = pd.read_csv(raw_dir / 'holidays_events.csv', parse_dates=['date'])
    oil = pd.read_csv(raw_dir / 'oil.csv', parse_dates=['date'])
    df = train.merge(tx, on=['date','store_nbr'], how='left').merge(oil, on='date', how='left')
    df['holiday_flag'] = df['date'].isin(set(holidays['date'])).astype(int)
    df['promotion_flag'] = df['onpromotion'].astype(int)
    df['week_id'] = ((df['date'] - df['date'].min()).dt.days // 7 + 1).astype(int)
    weekly = df.groupby(['store_nbr','item_nbr','week_id'], as_index=False).agg(
        units_actual=('unit_sales','sum'),
        promotion_flag=('promotion_flag','max'),
        holiday_flag=('holiday_flag','max'),
        transactions=('transactions','mean'),
        oil_price=('dcoilwtico','mean')
    )
    weekly['transaction_surge_flag'] = (weekly['transactions'] > weekly['transactions'].quantile(0.75)).astype(int)
    weekly['store_id'] = 'FAV_STORE_' + weekly['store_nbr'].astype(str)
    weekly['item_id'] = 'FAV_ITEM_' + weekly['item_nbr'].astype(str)
    weekly['category_id'] = 'FAVORITA'
    return weekly
