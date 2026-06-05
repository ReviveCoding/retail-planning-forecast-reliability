import pytest
from src.pipelines.run_all import run_all


def pytest_collection_modifyitems(config, items):
    for item in items:
        item.add_marker(pytest.mark.integration)


@pytest.fixture(scope='session', autouse=True)
def build_outputs_once_for_integration_tests():
    run_all()
    yield
