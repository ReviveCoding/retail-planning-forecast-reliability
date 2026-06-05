# Real Public Data Guide

This repository runs out of the box with generated sample M5-like and Favorita-like data. For public-data validation, place the public datasets manually under the following folders.

## M5 Forecasting Accuracy

Expected files:

```text
data/raw/m5/sales_train_validation.csv
data/raw/m5/calendar.csv
data/raw/m5/sell_prices.csv
```

The pipeline includes a subset guard in `configs/data_config.yml`:

```yaml
real_data_subset:
  enabled: true
  max_items: 250
  max_stores: 5
  max_days: 730
```

Keep this enabled for the first real-data run. Full M5 conversion can create a very large long-format table.

## Corporación Favorita Grocery Sales

Expected files are flexible because the current adapter supports sample fallback. Future full-public validation should use:

```text
data/raw/favorita/train.csv
data/raw/favorita/holidays_events.csv
data/raw/favorita/transactions.csv
data/raw/favorita/oil.csv
data/raw/favorita/items.csv
data/raw/favorita/stores.csv
```

## Claim boundary

Real public files do not make this an internal production system. Planning, inventory, capacity, and intervention variables remain synthetic/proxy assumptions unless a user explicitly provides real planning data.
