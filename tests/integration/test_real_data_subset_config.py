from src.utils import load_yaml


def test_real_data_subset_config_exists():
    cfg = load_yaml('configs/data_config.yml')
    assert cfg['real_data_subset']['enabled'] is True
    assert cfg['real_data_subset']['max_items'] > 0
    assert cfg['real_data_subset']['max_days'] > 0
