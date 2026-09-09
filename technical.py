"""
Technical indicators module.
Computes all 39 technical signals from OHLCV data.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

def _ema(series, span):
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series, window):
    """Simple moving average."""
    return series.rolling(window=window, min_periods=1).mean()


def _wma(series, window):
    """Weighted moving average."""
    weights = np.arange(1, window + 1, dtype=float)
    return series.rolling(window=window, min_periods=1).apply(
        lambda x: np.dot(x[-len(weights):], weights[-len(x):]) / weights[-len(x):].sum(),
        raw=True,
    )


def _true_range(high, low, close):
    """True Range calculation."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


# ═══════════════════════════════════════════════════════════════════
# 1. Bollinger Bands (5 indicators)
# ═══════════════════════════════════════════════════════════════════

def compute_bollinger_bands(close, window=20, num_std=2):
    """
    BB_MA, BB_UPPER, BB_LOWER, BB_WIDTH, BB_PERCENT
    """
    bb_ma = _sma(close, window)
    std = close.rolling(window=window, min_periods=1).std()
    bb_upper = bb_ma + num_std * std
    bb_lower = bb_ma - num_std * std
    bb_width = (bb_upper - bb_lower) / bb_ma
    bb_percent = (close - bb_lower) / (bb_upper - bb_lower)

    return {
        "BB_MA": bb_ma,
        "BB_UPPER": bb_upper,
        "BB_LOWER": bb_lower,
        "BB_WIDTH": bb_width,
        "BB_PERCENT": bb_percent,
    }


# ═══════════════════════════════════════════════════════════════════
# 2. Directional Movement (DMI_PLUS, DMI_MINUS, ADX, ADXR)
# ═══════════════════════════════════════════════════════════════════

def compute_dmi_adx(high, low, close, window=14):
    """
    DMI_PLUS, DMI_MINUS, ADX, ADXR
    """
    tr = _true_range(high, low, close)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    dm_plus = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    dm_minus = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    dm_plus = pd.Series(dm_plus, index=high.index)
    dm_minus = pd.Series(dm_minus, index=high.index)

    atr = _ema(tr, window)
    di_plus = 100 * _ema(dm_plus, window) / atr.replace(0, np.nan)
    di_minus = 100 * _ema(dm_minus, window) / atr.replace(0, np.nan)

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = _ema(dx, window)
    adxr = (adx + adx.shift(window)) / 2

    return {
        "DMI_PLUS": di_plus,
        "DMI_MINUS": di_minus,
        "ADX": adx,
        "ADXR": adxr,
    }


# ═══════════════════════════════════════════════════════════════════
# 3. Exponential Moving Average
# ═══════════════════════════════════════════════════════════════════

def compute_ema(close, span=12):
    """EMAVG: Exponential Moving Average."""
    return {"EMAVG": _ema(close, span)}


# ═══════════════════════════════════════════════════════════════════
# 4. Commodity Channel Index (CCI / CMCI)
# ═══════════════════════════════════════════════════════════════════

def compute_cci(high, low, close, window=20):
    """CMCI: Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = _sma(tp, window)
    mean_dev = tp.rolling(window=window, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))
    return {"CMCI": cci}


# ═══════════════════════════════════════════════════════════════════
# 5. Fear and Greed Index (approximation)
# ═══════════════════════════════════════════════════════════════════

def compute_fear_greed(close, high, low, volume, window=20):
    """
    FEAR_GREED: Composite indicator approximating market sentiment.
    Combines momentum, RSI-like measure, and volume trend.
    """
    # Momentum component
    mom = close.pct_change(window)

    # Strength component (simplified RSI)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_comp = 100 - 100 / (1 + rs)

    # Volume trend
    vol_trend = volume.rolling(window).mean() / volume.rolling(window * 3).mean()

    # Composite (normalized 0-100)
    fg = (rsi_comp * 0.4 + mom.rank(pct=True) * 100 * 0.3 +
          vol_trend.rank(pct=True) * 100 * 0.3)

    return {"FEAR_GREED": fg}


# ═══════════════════════════════════════════════════════════════════
# 6. Hurst Exponent
# ═══════════════════════════════════════════════════════════════════

def compute_hurst(close, max_lag=20):
    """HURST: Hurst exponent using R/S analysis."""
    log_returns = np.log(close / close.shift(1))

    def _hurst_rs(series):
        series = series.dropna()
        if len(series) < max_lag:
            return np.nan
        lags = range(2, min(max_lag + 1, len(series)))
        rs_list = []
        for lag in lags:
            subseries = series[:lag]
            mean_val = subseries.mean()
            devs = subseries - mean_val
            cumdev = devs.cumsum()
            R = cumdev.max() - cumdev.min()
            S = subseries.std()
            if S > 0:
                rs_list.append((lag, R / S))
        if len(rs_list) < 3:
            return np.nan
        log_lags = np.log([x[0] for x in rs_list])
        log_rs = np.log([x[1] for x in rs_list])
        hurst = np.polyfit(log_lags, log_rs, 1)[0]
        return hurst

    hurst = log_returns.rolling(window=max_lag * 2, min_periods=max_lag).apply(
        _hurst_rs, raw=False
    )
    return {"HURST": hurst}


# ═══════════════════════════════════════════════════════════════════
# 7. Min/Max and Retracement
# ═══════════════════════════════════════════════════════════════════

def compute_minmax(close, window=12):
    """MIN, MAX, MM_RETRACEMENT: 12-period min/max and retracement."""
    min_price = close.rolling(window=window, min_periods=1).min()
    max_price = close.rolling(window=window, min_periods=1).max()
    retracement = (close - min_price) / (max_price - min_price).replace(0, np.nan)

    return {
        "MIN": min_price,
        "MAX": max_price,
        "MM_RETRACEMENT": retracement,
    }


# ═══════════════════════════════════════════════════════════════════
# 8. Momentum + Momentum MA
# ═══════════════════════════════════════════════════════════════════

def compute_momentum(close, window=12, ma_window=9):
    """MOMENTUM, MOM_MA: Price momentum and its moving average."""
    momentum = close - close.shift(window)
    mom_ma = _sma(momentum, ma_window)
    return {"MOMENTUM": momentum, "MOM_MA": mom_ma}


# ═══════════════════════════════════════════════════════════════════
# 9. MACD (3 indicators)
# ═══════════════════════════════════════════════════════════════════

def compute_macd(close, fast=12, slow=26, signal=9):
    """MACD, MACD_SIGNAL, MACD_DIFF."""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd = ema_fast - ema_slow
    macd_signal = _ema(macd, signal)
    macd_diff = macd - macd_signal

    return {
        "MACD": macd,
        "MACD_SIGNAL": macd_signal,
        "MACD_DIFF": macd_diff,
    }


# ═══════════════════════════════════════════════════════════════════
# 10. Moving Average Envelopes
# ═══════════════════════════════════════════════════════════════════

def compute_ma_envelopes(close, window=20, pct=0.025):
    """MAEs: Moving Average Envelopes (% band around SMA)."""
    sma = _sma(close, window)
    maes = (close - sma) / (sma * pct).replace(0, np.nan)
    return {"MAEs": maes}


# ═══════════════════════════════════════════════════════════════════
# 11. Moving Average Oscillator (3 indicators)
# ═══════════════════════════════════════════════════════════════════

def compute_ma_oscillator(close, fast=10, slow=20, signal=9):
    """MAOsc, MAO_SIGNAL, MAO_DIFF."""
    sma_fast = _sma(close, fast)
    sma_slow = _sma(close, slow)
    ma_osc = sma_fast - sma_slow
    mao_signal = _ema(ma_osc, signal)
    mao_diff = ma_osc - mao_signal

    return {
        "MAOsc": ma_osc,
        "MAO_SIGNAL": mao_signal,
        "MAO_DIFF": mao_diff,
    }


# ═══════════════════════════════════════════════════════════════════
# 12. Rate of Change
# ═══════════════════════════════════════════════════════════════════

def compute_roc(close, window=12):
    """ROC: Rate of Change."""
    roc = (close / close.shift(window) - 1) * 100
    return {"ROC": roc}


# ═══════════════════════════════════════════════════════════════════
# 13. Parabolic SAR (simplified)
# ═══════════════════════════════════════════════════════════════════

def compute_parabolic_sar(high, low, close, af_start=0.02, af_step=0.02, af_max=0.20):
    """PTPS: Parabolic Time/Price System (Parabolic SAR)."""
    n = len(close)
    psar = np.zeros(n)
    af = af_start
    bull = True
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    psar[0] = close.iloc[0]

    for i in range(1, n):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
            psar[i] = min(psar[i], low.iloc[i - 1],
                          low.iloc[i - 2] if i >= 2 else low.iloc[i - 1])
            if low.iloc[i] < psar[i]:
                bull = False
                psar[i] = hp
                lp = low.iloc[i]
                af = af_start
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
            psar[i] = max(psar[i], high.iloc[i - 1],
                          high.iloc[i - 2] if i >= 2 else high.iloc[i - 1])
            if high.iloc[i] > psar[i]:
                bull = True
                psar[i] = lp
                hp = high.iloc[i]
                af = af_start
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + af_step, af_max)

    return {"PTPS": pd.Series(psar, index=close.index)}


# ═══════════════════════════════════════════════════════════════════
# 14. Relative Strength Index
# ═══════════════════════════════════════════════════════════════════

def compute_rsi(close, window=14):
    """RSI: Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = _ema(gain, window)
    avg_loss = _ema(loss, window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    return {"RSI": rsi}


# ═══════════════════════════════════════════════════════════════════
# 15. Simple Moving Average
# ═══════════════════════════════════════════════════════════════════

def compute_sma(close, window=20):
    """SMAVG: Simple Moving Average."""
    return {"SMAVG": _sma(close, window)}


# ═══════════════════════════════════════════════════════════════════
# 16. Stochastics Oscillator (4 indicators)
# ═══════════════════════════════════════════════════════════════════

def compute_stochastics(high, low, close, k_window=14, d_window=3, ds_window=3):
    """TAS_K, TAS_D, TAS_DS, TAS_DSS."""
    lowest_low = low.rolling(window=k_window, min_periods=1).min()
    highest_high = high.rolling(window=k_window, min_periods=1).max()

    tas_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    tas_d = _sma(tas_k, d_window)    # %D-Slow
    tas_ds = _sma(tas_d, ds_window)   # %DS-Slow
    tas_dss = _sma(tas_ds, ds_window) # MA of %DS-Slow

    return {
        "TAS_K": tas_k,
        "TAS_D": tas_d,
        "TAS_DS": tas_ds,
        "TAS_DSS": tas_dss,
    }


# ═══════════════════════════════════════════════════════════════════
# 17. Trender (Upper/Lower)
# ═══════════════════════════════════════════════════════════════════

def compute_trender(high, low, close, window=14, multiplier=2):
    """TRENDER_UP, TRENDER_DN: Upper and Lower Trender values."""
    atr = _true_range(high, low, close).rolling(window=window, min_periods=1).mean()
    mid = (high + low) / 2

    trender_up = mid + multiplier * atr
    trender_dn = mid - multiplier * atr

    return {
        "TRENDER_UP": trender_up,
        "TRENDER_DN": trender_dn,
    }


# ═══════════════════════════════════════════════════════════════════
# 18. Triangular Moving Average
# ═══════════════════════════════════════════════════════════════════

def compute_triangular_ma(close, window=20):
    """TMAVG: Triangular Moving Average (double-smoothed SMA)."""
    half = (window + 1) // 2
    sma1 = _sma(close, half)
    tmavg = _sma(sma1, half)
    return {"TMAVG": tmavg}


# ═══════════════════════════════════════════════════════════════════
# 19. Variable Moving Average
# ═══════════════════════════════════════════════════════════════════

def compute_variable_ma(close, window=20):
    """VMAVG: Variable Moving Average using CMO-based smoothing."""
    # Chande Momentum Oscillator as dynamic factor
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).sum()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).sum()
    cmo = ((gain - loss) / (gain + loss).replace(0, np.nan)).abs()

    # VMA: use CMO as scaling factor for EMA-like smoothing
    sc = 2.0 / (window + 1) * cmo
    vma = pd.Series(np.nan, index=close.index)
    vma.iloc[window - 1] = close.iloc[:window].mean()

    for i in range(window, len(close)):
        vma.iloc[i] = sc.iloc[i] * close.iloc[i] + (1 - sc.iloc[i]) * vma.iloc[i - 1]

    return {"VMAVG": vma}


# ═══════════════════════════════════════════════════════════════════
# 20. Weighted Moving Average
# ═══════════════════════════════════════════════════════════════════

def compute_weighted_ma(close, window=20):
    """WMAVG: Weighted Moving Average."""
    return {"WMAVG": _wma(close, window)}


# ═══════════════════════════════════════════════════════════════════
# 21. Williams %R
# ═══════════════════════════════════════════════════════════════════

def compute_williams_r(high, low, close, window=14):
    """WLPR: Williams' %R."""
    highest_high = high.rolling(window=window, min_periods=1).max()
    lowest_low = low.rolling(window=window, min_periods=1).min()
    wlpr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return {"WLPR": wlpr}


# ═══════════════════════════════════════════════════════════════════
# Master: Compute all 39 indicators for a single stock
# ═══════════════════════════════════════════════════════════════════

def compute_all_indicators(ohlcv_df):
    """
    Compute all 39 technical indicators for a single stock.

    Parameters
    ----------
    ohlcv_df : DataFrame with columns [Open, High, Low, Close, Volume]
               indexed by date.

    Returns
    -------
    dict of {indicator_name: Series}
    """
    o = ohlcv_df["Open"]
    h = ohlcv_df["High"]
    l = ohlcv_df["Low"]
    c = ohlcv_df["Close"]
    v = ohlcv_df["Volume"]

    indicators = {}
    indicators.update(compute_bollinger_bands(c))
    indicators.update(compute_dmi_adx(h, l, c))
    indicators.update(compute_ema(c))
    indicators.update(compute_cci(h, l, c))
    indicators.update(compute_fear_greed(c, h, l, v))
    indicators.update(compute_hurst(c))
    indicators.update(compute_minmax(c))
    indicators.update(compute_momentum(c))
    indicators.update(compute_macd(c))
    indicators.update(compute_ma_envelopes(c))
    indicators.update(compute_ma_oscillator(c))
    indicators.update(compute_roc(c))
    indicators.update(compute_parabolic_sar(h, l, c))
    indicators.update(compute_rsi(c))
    indicators.update(compute_sma(c))
    indicators.update(compute_stochastics(h, l, c))
    indicators.update(compute_trender(h, l, c))
    indicators.update(compute_triangular_ma(c))
    indicators.update(compute_variable_ma(c))
    indicators.update(compute_weighted_ma(c))
    indicators.update(compute_williams_r(h, l, c))

    return indicators


def compute_technical_indicators_panel(prices_df):
    """
    Compute all 39 technical indicators for all stocks.

    Parameters
    ----------
    prices_df : DataFrame with columns [date, ticker, Open, High, Low, Close, Volume].

    Returns
    -------
    ti_df : DataFrame indexed like prices_df with 39 TI columns.
    """
    print("[FEATURES] Computing 39 technical indicators for all stocks...")

    tickers = prices_df["ticker"].unique()
    all_results = []

    for ticker in tqdm(tickers, desc="Computing TIs"):
        stock = prices_df[prices_df["ticker"] == ticker].copy()
        stock = stock.sort_values("date").set_index("date")

        if len(stock) < 5:
            continue

        try:
            indicators = compute_all_indicators(stock)
            ti_row = pd.DataFrame(indicators, index=stock.index)
            ti_row["ticker"] = ticker
            all_results.append(ti_row.reset_index())
        except Exception as e:
            print(f"  → Warning: Failed for {ticker}: {e}")
            continue

    ti_df = pd.concat(all_results, ignore_index=True)

    actual = [c for c in config.TECHNICAL_INDICATORS if c in ti_df.columns]
    print(f"  → Computed {len(actual)}/39 technical indicators for "
          f"{len(tickers)} stocks")

    return ti_df
