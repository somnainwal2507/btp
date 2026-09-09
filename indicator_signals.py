"""
Indicator signal generation module.
Converts each of the 39 technical indicators into binary buy/sell signals.

Signal convention:
    1 = BUY  (go long / stay invested)
    0 = SELL (go to cash / exit)
    NaN = no signal yet (insufficient data)
"""

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_system import signal_config as sc


# ═══════════════════════════════════════════════════════════════════
# Signal Generation Functions
# ═══════════════════════════════════════════════════════════════════

def _trend_signal(close, indicator_values):
    """
    Trend-following signal: BUY if price > indicator, SELL if price < indicator.
    Used for moving averages (SMA, EMA, WMA, TMA, VMA, BB_MA).
    """
    signal = pd.Series(np.nan, index=close.index)
    signal[close > indicator_values] = 1.0
    signal[close < indicator_values] = 0.0
    return signal.ffill()


def _oscillator_signal(values, oversold, overbought):
    """
    Contrarian oscillator signal:
        BUY  if value < oversold  (extreme low → reversal expected)
        SELL if value > overbought (extreme high → reversal expected)
        HOLD previous signal in the neutral zone
    """
    signal = pd.Series(np.nan, index=values.index)
    signal[values < oversold] = 1.0
    signal[values > overbought] = 0.0
    return signal.ffill()


def _crossover_signal(fast, slow):
    """
    Crossover signal: BUY if fast > slow, SELL if fast < slow.
    Used for MACD vs Signal, DMI+ vs DMI-, etc.
    """
    signal = pd.Series(np.nan, index=fast.index)
    signal[fast > slow] = 1.0
    signal[fast < slow] = 0.0
    return signal.ffill()


def _momentum_signal(values):
    """
    Momentum signal: BUY if value > 0 (positive momentum), SELL if < 0.
    """
    signal = pd.Series(np.nan, index=values.index)
    signal[values > 0] = 1.0
    signal[values < 0] = 0.0
    return signal.ffill()


def _hurst_trend_signal(hurst_values, close):
    """
    Hurst-based signal:
        H > 0.5 (trending) → follow recent price direction
        H < 0.5 (mean-reverting) → expect reversal
    """
    signal = pd.Series(np.nan, index=close.index)
    price_direction = (close.diff() > 0).astype(float)

    trending = hurst_values > sc.HURST_TRENDING
    mean_reverting = hurst_values < sc.HURST_TRENDING

    # Trending → follow direction
    signal[trending & (price_direction == 1)] = 1.0
    signal[trending & (price_direction == 0)] = 0.0
    # Mean-reverting → go against direction
    signal[mean_reverting & (price_direction == 1)] = 0.0
    signal[mean_reverting & (price_direction == 0)] = 1.0

    return signal.ffill()


# ═══════════════════════════════════════════════════════════════════
# Master Signal Generator
# ═══════════════════════════════════════════════════════════════════

def generate_all_signals(indicator_df, close_prices):
    """
    Generate binary signals for all 39 technical indicators.

    Parameters
    ----------
    indicator_df : DataFrame
        Contains columns for each of the 39 technical indicators,
        indexed in the same order as close_prices.
    close_prices : Series
        S&P 500 closing prices, same index as indicator_df.

    Returns
    -------
    signals_df : DataFrame
        Binary signals (0/1) for each indicator, same index.
        Column names match indicator names.
    """
    close = close_prices.copy()
    signals = {}

    # ─── 1. Moving Averages (Trend-Following) ───
    for ma_name in ["SMAVG", "EMAVG", "WMAVG", "TMAVG", "VMAVG", "BB_MA"]:
        if ma_name in indicator_df.columns:
            signals[ma_name] = _trend_signal(close, indicator_df[ma_name])

    # ─── 2. Bollinger Bands ───
    if "BB_UPPER" in indicator_df.columns:
        # Buy near lower band, sell near upper band
        signals["BB_UPPER"] = _oscillator_signal(
            close,
            oversold=indicator_df["BB_LOWER"],
            overbought=indicator_df["BB_UPPER"],
        )

    if "BB_LOWER" in indicator_df.columns:
        # Redundant with BB_UPPER signal, use BB_PERCENT instead
        signals["BB_LOWER"] = _oscillator_signal(
            close,
            oversold=indicator_df["BB_LOWER"],
            overbought=indicator_df["BB_UPPER"],
        )

    if "BB_WIDTH" in indicator_df.columns:
        # Wide bands = high volatility → contrarian (expect mean reversion)
        bb_width_median = indicator_df["BB_WIDTH"].expanding(min_periods=12).median()
        signal = pd.Series(np.nan, index=close.index)
        signal[indicator_df["BB_WIDTH"] > bb_width_median * 1.5] = 1.0   # High vol → buy (contrarian)
        signal[indicator_df["BB_WIDTH"] < bb_width_median * 0.5] = 0.0   # Low vol → sell (expect expansion)
        signals["BB_WIDTH"] = signal.ffill()

    if "BB_PERCENT" in indicator_df.columns:
        signals["BB_PERCENT"] = _oscillator_signal(
            indicator_df["BB_PERCENT"],
            oversold=sc.BB_PERCENT_LOW,
            overbought=sc.BB_PERCENT_HIGH,
        )

    # ─── 3. DMI / ADX ───
    if "DMI_PLUS" in indicator_df.columns and "DMI_MINUS" in indicator_df.columns:
        signals["DMI_PLUS"] = _crossover_signal(
            indicator_df["DMI_PLUS"], indicator_df["DMI_MINUS"]
        )
        # DMI_MINUS: inverse of DMI_PLUS
        signals["DMI_MINUS"] = _crossover_signal(
            indicator_df["DMI_MINUS"], indicator_df["DMI_PLUS"]
        )

    if "ADX" in indicator_df.columns:
        # ADX > threshold → strong trend → follow DMI direction
        adx = indicator_df["ADX"]
        dmi_signal = signals.get("DMI_PLUS", pd.Series(np.nan, index=close.index))
        signal = pd.Series(np.nan, index=close.index)
        signal[adx > sc.ADX_TREND_THRESHOLD] = dmi_signal[adx > sc.ADX_TREND_THRESHOLD]
        signal[adx <= sc.ADX_TREND_THRESHOLD] = 0.5  # Neutral when no trend
        signals["ADX"] = signal.ffill()

    if "ADXR" in indicator_df.columns:
        # ADXR is smoothed ADX, same logic
        adxr = indicator_df["ADXR"]
        dmi_signal = signals.get("DMI_PLUS", pd.Series(np.nan, index=close.index))
        signal = pd.Series(np.nan, index=close.index)
        signal[adxr > sc.ADX_TREND_THRESHOLD] = dmi_signal[adxr > sc.ADX_TREND_THRESHOLD]
        signal[adxr <= sc.ADX_TREND_THRESHOLD] = 0.5
        signals["ADXR"] = signal.ffill()

    # ─── 4. RSI ───
    if "RSI" in indicator_df.columns:
        signals["RSI"] = _oscillator_signal(
            indicator_df["RSI"],
            oversold=sc.RSI_OVERSOLD,
            overbought=sc.RSI_OVERBOUGHT,
        )

    # ─── 5. MACD ───
    if "MACD" in indicator_df.columns and "MACD_SIGNAL" in indicator_df.columns:
        signals["MACD"] = _crossover_signal(
            indicator_df["MACD"], indicator_df["MACD_SIGNAL"]
        )

    if "MACD_SIGNAL" in indicator_df.columns:
        # Signal line direction
        signals["MACD_SIGNAL"] = _momentum_signal(indicator_df["MACD_SIGNAL"])

    if "MACD_DIFF" in indicator_df.columns:
        signals["MACD_DIFF"] = _momentum_signal(indicator_df["MACD_DIFF"])

    # ─── 6. Stochastics ───
    if "TAS_K" in indicator_df.columns:
        signals["TAS_K"] = _oscillator_signal(
            indicator_df["TAS_K"],
            oversold=sc.STOCH_OVERSOLD,
            overbought=sc.STOCH_OVERBOUGHT,
        )

    if "TAS_D" in indicator_df.columns:
        signals["TAS_D"] = _oscillator_signal(
            indicator_df["TAS_D"],
            oversold=sc.STOCH_OVERSOLD,
            overbought=sc.STOCH_OVERBOUGHT,
        )

    if "TAS_DS" in indicator_df.columns:
        signals["TAS_DS"] = _oscillator_signal(
            indicator_df["TAS_DS"],
            oversold=sc.STOCH_OVERSOLD,
            overbought=sc.STOCH_OVERBOUGHT,
        )

    if "TAS_DSS" in indicator_df.columns:
        signals["TAS_DSS"] = _oscillator_signal(
            indicator_df["TAS_DSS"],
            oversold=sc.STOCH_OVERSOLD,
            overbought=sc.STOCH_OVERBOUGHT,
        )

    # ─── 7. Momentum / ROC ───
    if "MOMENTUM" in indicator_df.columns:
        signals["MOMENTUM"] = _momentum_signal(indicator_df["MOMENTUM"])

    if "MOM_MA" in indicator_df.columns:
        signals["MOM_MA"] = _momentum_signal(indicator_df["MOM_MA"])

    if "ROC" in indicator_df.columns:
        signals["ROC"] = _momentum_signal(indicator_df["ROC"])

    # ─── 8. MA Oscillator ───
    if "MAOsc" in indicator_df.columns:
        signals["MAOsc"] = _momentum_signal(indicator_df["MAOsc"])

    if "MAO_SIGNAL" in indicator_df.columns:
        signals["MAO_SIGNAL"] = _momentum_signal(indicator_df["MAO_SIGNAL"])

    if "MAO_DIFF" in indicator_df.columns:
        signals["MAO_DIFF"] = _momentum_signal(indicator_df["MAO_DIFF"])

    # ─── 9. MA Envelopes ───
    if "MAEs" in indicator_df.columns:
        signals["MAEs"] = _oscillator_signal(
            indicator_df["MAEs"],
            oversold=sc.MA_ENVELOPE_LOW,
            overbought=sc.MA_ENVELOPE_HIGH,
        )

    # ─── 10. Parabolic SAR ───
    if "PTPS" in indicator_df.columns:
        signals["PTPS"] = _trend_signal(close, indicator_df["PTPS"])

    # ─── 11. Williams %R ───
    if "WLPR" in indicator_df.columns:
        signals["WLPR"] = _oscillator_signal(
            indicator_df["WLPR"],
            oversold=sc.WILLIAMS_OVERSOLD,
            overbought=sc.WILLIAMS_OVERBOUGHT,
        )

    # ─── 12. Trender ───
    if "TRENDER_UP" in indicator_df.columns:
        # Buy if price breaks above upper trender
        signal = pd.Series(np.nan, index=close.index)
        signal[close > indicator_df["TRENDER_UP"]] = 1.0
        signal[close < indicator_df["TRENDER_DN"]] = 0.0
        signals["TRENDER_UP"] = signal.ffill()

    if "TRENDER_DN" in indicator_df.columns:
        # Same logic from lower perspective
        signal = pd.Series(np.nan, index=close.index)
        signal[close > indicator_df["TRENDER_DN"]] = 1.0
        signal[close < indicator_df["TRENDER_DN"]] = 0.0
        signals["TRENDER_DN"] = signal.ffill()

    # ─── 13. Min / Max / Retracement ───
    if "MIN" in indicator_df.columns:
        # Near rolling min → buy (support bounce)
        pct_from_min = (close - indicator_df["MIN"]) / close.replace(0, np.nan)
        signal = pd.Series(np.nan, index=close.index)
        signal[pct_from_min < 0.02] = 1.0   # Within 2% of min → buy
        signal[pct_from_min > 0.10] = 0.0   # Far from min → neutral/sell
        signals["MIN"] = signal.ffill()

    if "MAX" in indicator_df.columns:
        # Near rolling max → sell (resistance rejection)
        pct_from_max = (indicator_df["MAX"] - close) / close.replace(0, np.nan)
        signal = pd.Series(np.nan, index=close.index)
        signal[pct_from_max < 0.02] = 0.0   # At max → sell
        signal[pct_from_max > 0.10] = 1.0   # Far from max → room to grow
        signals["MAX"] = signal.ffill()

    if "MM_RETRACEMENT" in indicator_df.columns:
        signals["MM_RETRACEMENT"] = _oscillator_signal(
            indicator_df["MM_RETRACEMENT"],
            oversold=sc.RETRACEMENT_LOW,
            overbought=sc.RETRACEMENT_HIGH,
        )

    # ─── 14. CCI (CMCI) ───
    if "CMCI" in indicator_df.columns:
        signals["CMCI"] = _oscillator_signal(
            indicator_df["CMCI"],
            oversold=sc.CCI_OVERSOLD,
            overbought=sc.CCI_OVERBOUGHT,
        )

    # ─── 15. Fear & Greed ───
    if "FEAR_GREED" in indicator_df.columns:
        signals["FEAR_GREED"] = _oscillator_signal(
            indicator_df["FEAR_GREED"],
            oversold=sc.FEAR_GREED_FEAR,
            overbought=sc.FEAR_GREED_GREED,
        )

    # ─── 16. Hurst Exponent ───
    if "HURST" in indicator_df.columns:
        signals["HURST"] = _hurst_trend_signal(indicator_df["HURST"], close)

    # Build DataFrame
    signals_df = pd.DataFrame(signals, index=close.index)

    # Fill initial NaNs with 0.5 (neutral)
    signals_df = signals_df.fillna(0.5)

    # Clip to [0, 1]
    signals_df = signals_df.clip(0, 1)

    return signals_df


def get_signal_descriptions():
    """
    Return a dict mapping indicator names to human-readable signal descriptions.
    Used for documentation and visualization.
    """
    return {
        "SMAVG":  "Price > SMA(20) → BUY",
        "EMAVG":  "Price > EMA(12) → BUY",
        "WMAVG":  "Price > WMA(20) → BUY",
        "TMAVG":  "Price > TMA(20) → BUY",
        "VMAVG":  "Price > VMA(20) → BUY",
        "BB_MA":  "Price > BB Middle Band → BUY",
        "BB_UPPER": "Price < BB Lower → BUY, > BB Upper → SELL",
        "BB_LOWER": "Price < BB Lower → BUY, > BB Upper → SELL",
        "BB_WIDTH": "High BB Width → contrarian BUY",
        "BB_PERCENT": "BB%B < 0.2 → BUY, > 0.8 → SELL",
        "DMI_PLUS": "DMI+ > DMI- → BUY",
        "DMI_MINUS": "DMI- > DMI+ → BUY (bearish)",
        "ADX":    "ADX > 25 → follow DMI direction",
        "ADXR":   "ADXR > 25 → follow DMI direction",
        "RSI":    "RSI < 30 → BUY (oversold), > 70 → SELL",
        "MACD":   "MACD > Signal line → BUY",
        "MACD_SIGNAL": "Signal line > 0 → BUY",
        "MACD_DIFF": "MACD histogram > 0 → BUY",
        "TAS_K":  "%K < 20 → BUY (oversold), > 80 → SELL",
        "TAS_D":  "%D < 20 → BUY, > 80 → SELL",
        "TAS_DS": "%DS < 20 → BUY, > 80 → SELL",
        "TAS_DSS": "%DSS < 20 → BUY, > 80 → SELL",
        "MOMENTUM": "Momentum > 0 → BUY",
        "MOM_MA": "Momentum MA > 0 → BUY",
        "ROC":    "ROC > 0 → BUY",
        "MAOsc":  "MA Oscillator > 0 → BUY",
        "MAO_SIGNAL": "MAO Signal > 0 → BUY",
        "MAO_DIFF": "MAO Histogram > 0 → BUY",
        "MAEs":   "MA Envelope < -1 → BUY, > 1 → SELL",
        "PTPS":   "Price > Parabolic SAR → BUY",
        "WLPR":   "Williams %R < -80 → BUY, > -20 → SELL",
        "TRENDER_UP": "Price > Upper Trender → BUY",
        "TRENDER_DN": "Price > Lower Trender → BUY",
        "MIN":    "Near rolling min → BUY (support)",
        "MAX":    "Near rolling max → SELL (resistance)",
        "MM_RETRACEMENT": "Retracement < 0.3 → BUY, > 0.7 → SELL",
        "CMCI":   "CCI < -100 → BUY, > 100 → SELL",
        "FEAR_GREED": "FG < 30 → BUY (fear), > 70 → SELL (greed)",
        "HURST":  "H > 0.5 → trend-follow, H < 0.5 → mean-revert",
    }
