"""
Dynamic weight optimization for technical indicator signals.
Calculates accuracy-based weights using expanding historical windows,
with optional volatility-adaptive adjustments.
"""

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_system import signal_config as sc


def compute_signal_accuracy(signals_df, actual_returns, indicator_names=None):
    """
    Calculate directional accuracy of each indicator's signals.

    Accuracy = proportion of months where the signal correctly predicted
    the direction of the next month's return.
        - Signal = 1 (BUY) and return > 0 → correct
        - Signal = 0 (SELL) and return <= 0 → correct
        - Signal = 0.5 (neutral) → counted as 50% correct

    Parameters
    ----------
    signals_df : DataFrame
        Binary signals (0/1) for each indicator.
    actual_returns : Series
        Actual next-month returns (shifted to match signal timing).
    indicator_names : list, optional
        Subset of indicators to evaluate. Default: all columns.

    Returns
    -------
    accuracy : Series
        Accuracy for each indicator (0.0 to 1.0).
    """
    if indicator_names is None:
        indicator_names = signals_df.columns.tolist()

    # Align lengths
    min_len = min(len(signals_df), len(actual_returns))
    sigs = signals_df.iloc[:min_len][indicator_names].copy()
    rets = actual_returns.iloc[:min_len].values

    # Direction: 1 if return > 0, 0 if return <= 0
    actual_direction = (rets > 0).astype(float)

    accuracies = {}
    for col in indicator_names:
        sig = sigs[col].values
        # Mask out neutral signals (0.5)
        non_neutral = np.abs(sig - 0.5) > 0.01  # not exactly 0.5
        if non_neutral.sum() < 5:
            # Too few active signals → default accuracy
            accuracies[col] = 0.5
            continue

        # Binary signals: round to 0 or 1
        binary_sig = (sig[non_neutral] > 0.5).astype(float)
        actual_dir = actual_direction[non_neutral]

        correct = (binary_sig == actual_dir).sum()
        total = len(binary_sig)
        accuracies[col] = correct / total

    return pd.Series(accuracies)


def compute_weights(accuracies, prev_weights=None, index_returns=None,
                    is_first_iteration=False):
    """
    Compute normalized weights from signal accuracies.

    Steps:
    1. Apply exponential smoothing with previous weights
    2. Apply SHAP-informed initial boost (first iteration only)
    3. Apply volatility-adaptive adjustment (optional)
    4. Apply softmax normalization
    5. Enforce minimum weight floor

    Parameters
    ----------
    accuracies : Series
        Signal accuracy for each indicator.
    prev_weights : Series, optional
        Previous iteration's weights for exponential smoothing.
    index_returns : Series, optional
        Recent index returns for volatility calculation.
    is_first_iteration : bool
        If True, apply SHAP-based initial weight boost.

    Returns
    -------
    weights : Series
        Normalized weights summing to 1.0.
    """
    indicators = accuracies.index.tolist()
    acc = accuracies.copy()

    # ─── Step 1: Exponential Smoothing ───
    if prev_weights is not None:
        alpha = sc.WEIGHT_SMOOTHING_ALPHA
        # Smooth the accuracies (not the weights directly)
        prev_acc = prev_weights.copy()  # Previous weights reflect previous accuracy
        for ind in indicators:
            if ind in prev_acc.index:
                acc[ind] = alpha * acc[ind] + (1 - alpha) * prev_acc[ind]

    # ─── Step 2: SHAP-Informed Initial Boost ───
    if is_first_iteration:
        for ind in sc.HIGH_SHAP_INDICATORS:
            if ind in acc.index:
                acc[ind] = acc[ind] * sc.INITIAL_WEIGHT_BOOST

    # ─── Step 3: Volatility-Adaptive Adjustment ───
    if sc.ENABLE_VOL_ADAPTIVE and index_returns is not None:
        vol_adjustment = _compute_vol_adjustment(index_returns)
        for ind in indicators:
            if ind in sc.OSCILLATOR_INDICATORS:
                acc[ind] *= vol_adjustment["oscillator_factor"]
            elif ind in sc.TREND_INDICATORS:
                acc[ind] *= vol_adjustment["trend_factor"]

    # ─── Step 4: Softmax Normalization ───
    # Shift accuracies to center around 0 for softmax stability
    centered = acc - acc.mean()
    exp_vals = np.exp(centered / sc.SOFTMAX_TEMPERATURE)
    weights = exp_vals / exp_vals.sum()

    # ─── Step 4.5: Win-Rate Penalty ───
    # If historical accuracy < threshold, assign 0 weight
    penalty_mask = accuracies < sc.MIN_HISTORICAL_WIN_RATE
    weights[penalty_mask] = 0.0
    if weights.sum() == 0:
        # Fallback if all indicators perform poorly
        weights = pd.Series(1.0 / len(weights), index=weights.index)
        penalty_mask[:] = False

    # ─── Step 5: Minimum Weight Floor ───
    weights = weights.clip(lower=sc.MIN_WEIGHT_FLOOR)
    # Re-zero the penalized ones
    weights[penalty_mask] = 0.0
    
    weights = weights / weights.sum()  # Re-normalize

    return weights


def _compute_vol_adjustment(returns):
    """
    Determine if we're in a high-volatility regime and adjust factors.

    Parameters
    ----------
    returns : Series
        Recent index returns.

    Returns
    -------
    dict with 'oscillator_factor' and 'trend_factor'.
    """
    if len(returns) < sc.VOL_LOOKBACK:
        return {"oscillator_factor": 1.0, "trend_factor": 1.0}

    recent_vol = returns.iloc[-sc.VOL_LOOKBACK:].std()
    historical_vol = returns.std()
    historical_mean_vol = returns.rolling(sc.VOL_LOOKBACK).std().mean()

    if historical_mean_vol == 0 or np.isnan(historical_mean_vol):
        return {"oscillator_factor": 1.0, "trend_factor": 1.0}

    vol_z_score = (recent_vol - historical_mean_vol) / returns.rolling(
        sc.VOL_LOOKBACK).std().std()

    if np.isnan(vol_z_score):
        return {"oscillator_factor": 1.0, "trend_factor": 1.0}

    if vol_z_score > sc.VOL_HIGH_THRESHOLD:
        # High volatility → boost oscillators, reduce trend-following
        return {
            "oscillator_factor": sc.OSCILLATOR_BOOST_FACTOR,
            "trend_factor": sc.TREND_REDUCE_FACTOR,
        }
    else:
        return {"oscillator_factor": 1.0, "trend_factor": 1.0}


def compute_weight_history(signals_df, actual_returns, test_years):
    """
    Compute weights for each test year using expanding window.

    Parameters
    ----------
    signals_df : DataFrame
        Full signal matrix (all years).
    actual_returns : Series
        Full return series with year information.
    test_years : list
        List of test years.

    Returns
    -------
    weight_history : dict of {year: Series of weights}
    """
    weight_history = {}
    prev_weights = None

    for i, test_year in enumerate(test_years):
        # Training data: everything before test_year
        train_mask = actual_returns.index < pd.Timestamp(f"{test_year}-01-01")
        train_signals = signals_df.loc[train_mask]
        train_returns = actual_returns.loc[train_mask]

        if len(train_signals) < 12:
            # Not enough data, use uniform weights
            n_indicators = len(signals_df.columns)
            weight_history[test_year] = pd.Series(
                1.0 / n_indicators, index=signals_df.columns
            )
            continue

        # Compute accuracies
        accuracies = compute_signal_accuracy(train_signals, train_returns)

        # Compute weights
        weights = compute_weights(
            accuracies,
            prev_weights=prev_weights,
            index_returns=train_returns,
            is_first_iteration=(i == 0),
        )

        weight_history[test_year] = weights
        prev_weights = weights.copy()

    return weight_history
