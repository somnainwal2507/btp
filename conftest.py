"""Shared fixtures for BTP test suite."""
import os
import sys

# Add project root to path for test discovery
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def price_series():
    """Simple ascending price series (100 daily points)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.linspace(100, 150, 100), index=idx)


@pytest.fixture
def flat_price():
    """Flat price series (edge case)."""
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.Series(100.0, index=idx)


@pytest.fixture
def sample_signals():
    """Binary signal DataFrame for 3 indicators."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    return pd.DataFrame({
        "RSI": np.random.choice([0.0, 1.0], 20),
        "MACD": np.random.choice([0.0, 1.0], 20),
        "SMAVG": np.random.choice([0.0, 1.0], 20),
    }, index=idx)


@pytest.fixture
def sample_returns():
    """Random return series."""
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=20, freq="D")
    return pd.Series(np.random.normal(0.001, 0.02, 20), index=idx)
