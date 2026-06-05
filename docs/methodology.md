# Methodology

## Forecasting setup

The pipeline creates an item-store-week panel from real M5 files when available, otherwise it generates sample M5-like retail demand data. It evaluates seasonal naive, moving-average, and tree-based ML baselines over 1-week and 4-week horizons.

## Synthetic plan setup

The plan layer is intentionally synthetic. It converts baseline forecasts into planning assumptions using growth, event, promotion, strategic override, safety-stock, and capacity-buffer factors.

## Variance bridge

Plan-vs-Actual revenue gap is decomposed into unit, price, interaction, inventory-proxy, capacity-proxy, and residual effects. The residual term preserves accounting identity.

## Driver validation

Favorita-style data are used for promotion, holiday, and transaction-surge hypothesis validation. Methods include event study, DiD-style comparison, placebo tests, bootstrap intervals, and confidence scoring.

## Specialty modules

Demand-State Timestep adapts state-consistent timestep reasoning to retail demand. Business-mechanism residual correction separates structural expected demand from learned residuals. Asymmetric coverage metrics evaluate under-forecast coverage and over-buffer precision.

## Narrative evaluation

Planning narratives are checked for numeric consistency, driver faithfulness, caveat coverage, and unsupported-claim risk.
