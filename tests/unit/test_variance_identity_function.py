import pandas as pd
from src.variance_bridge.bridge_identity_tests import identity_passes


def test_identity_passes_on_exact_bridge_table():
    df = pd.DataFrame({
        'total_revenue_gap': [10.0],
        'unit_effect': [3.0],
        'price_effect': [2.0],
        'interaction_effect': [1.0],
        'inventory_proxy_effect': [1.5],
        'capacity_proxy_effect': [0.5],
        'residual_effect': [2.0],
        'identity_error': [0.0],
    })
    assert identity_passes(df, tolerance=1e-9)
