"""
Shared pytest fixtures.

All tests are designed to run fully offline (no network, no Zeek, no Mininet).
"""
import pytest
import numpy as np

from cps_defender.core.events import reset_bus
from cps_defender.core.models import AttackType
from cps_defender.testbed.traffic_sim import TrafficSimulator
from cps_defender.ids.feature_extractor import FeatureExtractor


@pytest.fixture(autouse=True)
def fresh_bus():
    """Isolate event bus state between tests."""
    reset_bus()
    yield
    reset_bus()


@pytest.fixture(scope="session")
def simulator():
    return TrafficSimulator(n_outstations=4, attack_probability=0.30, seed=0)


@pytest.fixture(scope="session")
def small_dataset(simulator):
    """200 flows — enough for fast unit tests."""
    return simulator.generate(n_flows=200)


@pytest.fixture(scope="session")
def large_dataset(simulator):
    """2000 flows — used for training accuracy tests."""
    return simulator.generate(n_flows=2000)


@pytest.fixture(scope="session")
def fitted_extractor(large_dataset):
    ext = FeatureExtractor()
    ext.fit(large_dataset)
    return ext


@pytest.fixture(scope="session")
def feature_matrix(large_dataset, fitted_extractor):
    return fitted_extractor.transform(large_dataset)


@pytest.fixture(scope="session")
def label_array(large_dataset):
    return np.array([f.label for f in large_dataset])
