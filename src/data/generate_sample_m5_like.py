from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_m5_like(out_dir: str | Path, n_items=20, n_stores=5, n_days=420, start_date='2023-01-01', seed=42) -> None:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(start_date, periods=n_days, freq='D')
    d_cols = [f'd_{i+1}' for i in range(n_days)]
    calendar = pd.DataFrame({'date': dates, 'd': d_cols})
    calendar['wm_yr_wk'] = (calendar.index // 7 + 1).astype(int)
    calendar['weekday'] = calendar['date'].dt.day_name()
    calendar['month'] = calendar['date'].dt.month
    calendar['event_name_1'] = np.where(calendar['date'].dt.dayofweek.isin([5, 6]), 'Weekend', '')
    holiday_idx = rng.choice(n_days, size=max(6, n_days // 45), replace=False)
    calendar.loc[holiday_idx, 'event_name_1'] = 'Holiday'
    calendar.to_csv(out_dir / 'calendar.csv', index=False)

    rows = []
    prices = []
    stores = [f'STORE_{s:02d}' for s in range(1, n_stores + 1)]
    states = ['CA', 'TX', 'WI', 'NY', 'NJ']
    cats = ['FOODS', 'HOUSEHOLD', 'HOBBIES']
    depts = ['FOODS_1', 'FOODS_2', 'HOUSEHOLD_1', 'HOBBIES_1']

    for item_idx in range(1, n_items + 1):
        cat = cats[item_idx % len(cats)]
        dept = depts[item_idx % len(depts)]
        item_id = f'ITEM_{item_idx:03d}'
        base_price = rng.uniform(2.0, 15.0)
        elasticity = rng.uniform(-0.5, -0.1)
        for store_idx, store_id in enumerate(stores):
            state_id = states[store_idx % len(states)]
            base = rng.uniform(8, 55) * (1.0 + 0.1 * store_idx)
            trend = rng.uniform(-0.005, 0.015)
            seasonal = 1 + 0.15 * np.sin(np.arange(n_days) * 2 * np.pi / 365 + rng.uniform(0, 2*np.pi))
            weekend = 1 + 0.18 * calendar['date'].dt.dayofweek.isin([5,6]).to_numpy()
            holiday = 1 + 0.35 * (calendar['event_name_1'].eq('Holiday')).to_numpy()
            noise = rng.normal(0, 0.12, n_days)
            price_series = base_price * (1 + 0.08*np.sin(np.arange(n_days)*2*np.pi/90) + rng.normal(0,0.015,n_days))
            price_factor = np.maximum(0.4, 1 + elasticity * (price_series/base_price - 1))
            lam = np.maximum(0.2, base * (1 + trend*np.arange(n_days)) * seasonal * weekend * holiday * price_factor * (1+noise))
            sales = rng.poisson(lam).astype(int)
            row = {
                'id': f'{item_id}_{store_id}_validation', 'item_id': item_id, 'dept_id': dept, 'cat_id': cat,
                'store_id': store_id, 'state_id': state_id
            }
            row.update({d: int(v) for d, v in zip(d_cols, sales)})
            rows.append(row)

            wk_price = pd.DataFrame({'wm_yr_wk': calendar['wm_yr_wk'], 'sell_price': price_series}).groupby('wm_yr_wk', as_index=False).mean()
            wk_price['store_id'] = store_id
            wk_price['item_id'] = item_id
            prices.append(wk_price)

    pd.DataFrame(rows).to_csv(out_dir / 'sales_train_validation.csv', index=False)
    pd.concat(prices, ignore_index=True).to_csv(out_dir / 'sell_prices.csv', index=False)
