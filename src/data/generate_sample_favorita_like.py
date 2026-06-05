from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_favorita_like(out_dir: str | Path, n_items=8, n_stores=3, n_days=210, start_date='2023-01-01', seed=7) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    holidays = pd.DataFrame({'date': dates[rng.choice(n_days, size=max(8, n_days//40), replace=False)], 'type': 'Holiday', 'locale': 'National'})
    holidays.to_csv(out_dir / 'holidays_events.csv', index=False)
    oil = pd.DataFrame({'date': dates, 'dcoilwtico': 70 + 8*np.sin(np.arange(n_days)*2*np.pi/90) + rng.normal(0, 2, n_days)})
    oil.to_csv(out_dir / 'oil.csv', index=False)
    stores = pd.DataFrame({'store_nbr': range(1, n_stores+1), 'city': [f'City{s}' for s in range(1,n_stores+1)], 'state': ['Pichincha','Guayas','Azuay','Manabi','Loja','El Oro'][:n_stores], 'type': ['A','B','C','D','A','B'][:n_stores], 'cluster': [1,2,3,4,2,1][:n_stores]})
    stores.to_csv(out_dir / 'stores.csv', index=False)
    items = pd.DataFrame({'item_nbr': range(1,n_items+1), 'family': [f'FAMILY_{i%5}' for i in range(1,n_items+1)], 'class': [i%4 for i in range(1,n_items+1)], 'perishable': [i%2 for i in range(1,n_items+1)]})
    items.to_csv(out_dir / 'items.csv', index=False)
    tx_rows = []
    train_rows = []
    holidays_set = set(holidays['date'])
    for store in range(1, n_stores+1):
        tx_base = rng.uniform(500, 2000)
        tx = tx_base * (1 + 0.2*np.sin(np.arange(n_days)*2*np.pi/30)) + rng.normal(0, 80, n_days)
        tx_rows.extend({'date': d, 'store_nbr': store, 'transactions': max(50, int(v))} for d, v in zip(dates, tx))
        for item in range(1, n_items+1):
            base = rng.uniform(5, 70)
            promo_days = rng.choice(n_days, size=n_days//9, replace=False)
            promo = np.zeros(n_days, dtype=int)
            promo[promo_days] = 1
            holiday_flag = np.array([d in holidays_set for d in dates])
            oil_effect = 1 + 0.02*(oil['dcoilwtico'].to_numpy() - oil['dcoilwtico'].mean()) / oil['dcoilwtico'].std()
            lam = base * (1 + 0.30*promo + 0.20*holiday_flag) * oil_effect * (1 + 0.1*np.sin(np.arange(n_days)*2*np.pi/365))
            units = rng.poisson(np.maximum(0.2, lam))
            for d, u, p in zip(dates, units, promo):
                train_rows.append({'date': d, 'store_nbr': store, 'item_nbr': item, 'unit_sales': float(u), 'onpromotion': bool(p)})
    pd.DataFrame(tx_rows).to_csv(out_dir / 'transactions.csv', index=False)
    pd.DataFrame(train_rows).to_csv(out_dir / 'train.csv', index=False)
