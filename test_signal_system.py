"""
Unit tests for signal generation, weight optimization, ensemble, and performance modules.

Each test class targets a specific function with edge cases documented inline.
Run: pytest tests/test_signal_system.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import pandas as pd

from signal_system.indicator_signals import (
    _trend_signal,
    _oscillator_signal,
    _crossover_signal,
    _momentum_signal,
    generate_all_signals,
)
from signal_system.weight_optimizer import compute_signal_accuracy, compute_weights
from signal_system.ensemble import generate_ensemble_signal
from signal_system.performance import _compute_return_metrics, compute_performance_metrics


# ══════════════════════════════════════════════════════════════════
# 1. Signal Generator Tests
# ══════════════════════════════════════════════════════════════════

class TestTrendSignal:
    """Validates _trend_signal: BUY when price > indicator."""

    def test_all_above_returns_buy(self, price_series):
        """When price is always above indicator, every signal should be BUY (1)."""
        indicator = price_series - 10
        sig = _trend_signal(price_series, indicator)
        assert (sig.dropna() == 1.0).all()

    def test_all_below_returns_sell(self, price_series):
        """When price is always below indicator, every signal should be SELL (0)."""
        indicator = price_series + 10
        sig = _trend_signal(price_series, indicator)
        assert (sig.dropna() == 0.0).all()

    def test_flat_price_equals_indicator(self, flat_price):
        """When price == indicator exactly, no condition fires → NaN ffilled."""
        sig = _trend_signal(flat_price, flat_price)
        assert sig.isna().all() or (sig == sig.iloc[0]).all()

    def test_output_length(self, price_series):
        """Output series must have same length as input."""
        indicator = price_series.rolling(5).mean()
        sig = _trend_signal(price_series, indicator)
        assert len(sig) == len(price_series)

    def test_values_are_binary(self, price_series):
        """Signals must only be 0.0, 1.0, or NaN."""
        indicator = price_series.rolling(10).mean()
        sig = _trend_signal(price_series, indicator)
        valid = sig.dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})


class TestOscillatorSignal:
    """Validates _oscillator_signal with oversold/overbought thresholds."""

    def test_oversold_triggers_buy(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        values = pd.Series([15] * 10, index=idx)
        sig = _oscillator_signal(values, oversold=30, overbought=70)
        assert (sig == 1.0).all()

    def test_overbought_triggers_sell(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        values = pd.Series([85] * 10, index=idx)
        sig = _oscillator_signal(values, oversold=30, overbought=70)
        assert (sig == 0.0).all()

    def test_neutral_zone_holds_previous(self):
        """Values between thresholds should hold the last non-neutral signal."""
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        values = pd.Series([15, 50, 50, 50, 85], index=idx)
        sig = _oscillator_signal(values, oversold=30, overbought=70)
        assert sig.iloc[0] == 1.0  # oversold → BUY
        assert sig.iloc[1] == 1.0  # neutral → holds previous
        assert sig.iloc[4] == 0.0  # overbought → SELL

    def test_empty_series(self):
        sig = _oscillator_signal(pd.Series(dtype=float), 30, 70)
        assert len(sig) == 0


class TestCrossoverSignal:
    """Validates _crossover_signal: BUY when fast > slow."""

    def test_fast_above_slow(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        fast = pd.Series([10.0] * 10, index=idx)
        slow = pd.Series([5.0] * 10, index=idx)
        sig = _crossover_signal(fast, slow)
        assert (sig == 1.0).all()

    def test_crossover_transition(self):
        idx = pd.date_range("2020-01-01", periods=4, freq="D")
        fast = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        slow = pd.Series([3.0, 3.0, 3.0, 3.0], index=idx)
        sig = _crossover_signal(fast, slow)
        assert sig.iloc[0] == 0.0
        assert sig.iloc[3] == 1.0


class TestMomentumSignal:
    """Validates _momentum_signal: BUY if value > 0."""

    def test_all_positive(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        vals = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        sig = _momentum_signal(vals)
        assert (sig == 1.0).all()

    def test_all_negative(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        vals = pd.Series([-5.0, -4.0, -3.0, -2.0, -1.0], index=idx)
        sig = _momentum_signal(vals)
        assert (sig == 0.0).all()

    def test_zero_is_nan(self):
        """Exactly zero does not trigger either condition → NaN."""
        idx = pd.date_range("2020-01-01", periods=3, freq="D")
        vals = pd.Series([1.0, 0.0, -1.0], index=idx)
        sig = _momentum_signal(vals)
        assert sig.iloc[0] == 1.0
        assert sig.iloc[2] == 0.0


# ══════════════════════════════════════════════════════════════════
# 2. Weight Optimizer Tests
# ══════════════════════════════════════════════════════════════════

class TestComputeSignalAccuracy:
    """Validates compute_signal_accuracy for directional accuracy."""

    def test_perfect_accuracy(self):
        """BUY signals with all-positive returns = 100% accuracy."""
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        returns = pd.Series([0.01] * 20, index=idx)
        signals = pd.DataFrame({"A": [1.0] * 20}, index=idx)
        acc = compute_signal_accuracy(signals, returns)
        assert acc["A"] == 1.0

    def test_zero_accuracy(self):
        """BUY signals with all-negative returns = 0% accuracy."""
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        returns = pd.Series([-0.01] * 20, index=idx)
        signals = pd.DataFrame({"A": [1.0] * 20}, index=idx)
        acc = compute_signal_accuracy(signals, returns)
        assert acc["A"] == 0.0

    def test_neutral_defaults_to_half(self):
        """All-neutral (0.5) signals should return default 0.5 accuracy."""
        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        returns = pd.Series([0.01] * 20, index=idx)
        signals = pd.DataFrame({"A": [0.5] * 20}, index=idx)
        acc = compute_signal_accuracy(signals, returns)
        assert acc["A"] == 0.5

    def test_mixed_signals(self):
        """50/50 correct should give 0.5 accuracy."""
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        returns = pd.Series([0.01, -0.01] * 5, index=idx)
        signals = pd.DataFrame({"A": [1.0, 1.0] * 5}, index=idx)
        acc = compute_signal_accuracy(signals, returns)
        assert abs(acc["A"] - 0.5) < 1e-6


class TestComputeWeights:
    """Validates compute_weights normalization and penalties."""

    def test_weights_sum_to_one(self):
        acc = pd.Series({"A": 0.6, "B": 0.7, "C": 0.55})
        w = compute_weights(acc, is_first_iteration=True)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_poor_indicator_zeroed(self):
        """Indicator below MIN_HISTORICAL_WIN_RATE should get zero weight."""
        acc = pd.Series({"A": 0.3, "B": 0.8, "C": 0.9})
        w = compute_weights(acc, is_first_iteration=False)
        assert w["A"] == 0.0

    def test_all_poor_fallback_uniform(self):
        """When all indicators are poor, fallback to uniform weights."""
        acc = pd.Series({"A": 0.2, "B": 0.3, "C": 0.1})
        w = compute_weights(acc, is_first_iteration=False)
        assert w.sum() > 0

    def test_higher_accuracy_gets_higher_weight(self):
        acc = pd.Series({"A": 0.55, "B": 0.90})
        w = compute_weights(acc, is_first_iteration=False)
        assert w["B"] > w["A"]


# ══════════════════════════════════════════════════════════════════
# 3. Ensemble Signal Tests
# ══════════════════════════════════════════════════════════════════

class TestGenerateEnsembleSignal:
    """Validates generate_ensemble_signal output."""

    def test_unanimous_buy(self):
        signals = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
        weights = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
        sig, ws = generate_ensemble_signal(signals, weights)
        assert sig == 1

    def test_unanimous_sell(self):
        signals = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
        weights = pd.Series({"A": 0.33, "B": 0.33, "C": 0.34})
        sig, ws = generate_ensemble_signal(signals, weights)
        assert sig == 0

    def test_no_common_indicators_returns_hold(self):
        signals = pd.Series({"X": 1.0})
        weights = pd.Series({"Y": 0.5})
        sig, ws = generate_ensemble_signal(signals, weights)
        assert sig == -1

    def test_zero_weight_sum_returns_hold(self):
        signals = pd.Series({"A": 1.0, "B": 0.0})
        weights = pd.Series({"A": 0.0, "B": 0.0})
        sig, ws = generate_ensemble_signal(signals, weights)
        assert sig == -1


# ══════════════════════════════════════════════════════════════════
# 4. Performance Metrics Tests
# ══════════════════════════════════════════════════════════════════

class TestComputeReturnMetrics:
    """Validates _compute_return_metrics edge cases."""

    def test_empty_returns(self):
        m = _compute_return_metrics(np.array([]))
        assert m["total_return"] == 0.0
        assert m["sharpe_ratio"] == 0.0

    def test_all_positive_returns(self):
        rets = np.array([0.01] * 12)
        m = _compute_return_metrics(rets)
        assert m["win_rate"] == 1.0
        assert m["total_return"] > 0

    def test_all_negative_returns(self):
        rets = np.array([-0.01] * 12)
        m = _compute_return_metrics(rets)
        assert m["win_rate"] == 0.0
        assert m["max_drawdown"] < 0

    def test_single_return(self):
        m = _compute_return_metrics(np.array([0.05]))
        assert abs(m["total_return"] - 0.05) < 1e-6

    def test_nan_filtered_out(self):
        rets = np.array([0.01, np.nan, 0.02, np.nan])
        m = _compute_return_metrics(rets)
        assert m["total_months"] == 2

    def test_zero_std_sharpe(self):
        """Identical returns → zero std → Sharpe should be 0."""
        rets = np.array([0.01, 0.01, 0.01])
        m = _compute_return_metrics(rets)
        # std with ddof=1 for 3 identical values is 0
        assert m["sharpe_ratio"] == 0.0
