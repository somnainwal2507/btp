"""
Configuration for the Technical Indicator Signal System.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# ─────────────────────────── Paths ───────────────────────────
SIGNAL_RESULTS_DIR = os.path.join(config.RESULTS_DIR, "signal_system")
os.makedirs(SIGNAL_RESULTS_DIR, exist_ok=True)
INDEX_CACHE_FILE = "sp500_index.parquet"

# ─────────────────────────── Backtest Sizes (Row counts, timeframe-agnostic) ───────────────────────────
TRAIN_PERIODS = 120
TEST_PERIODS = 12

# ─────────────────────────── Indicator Lookback Parameters (Adaptive) ───────────────────────────
# Base periods for indicators. These should scale logically depending on timeframe.
LOOKBACK_SHORT = 14
LOOKBACK_MEDIUM = 20
LOOKBACK_LONG = 50

# ─────────────────────────── Signal Thresholds ───────────────────────────
# Oscillator thresholds (contrarian signals)
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

WILLIAMS_OVERSOLD = -80       # Williams %R scale: 0 to -100
WILLIAMS_OVERBOUGHT = -20

CCI_OVERSOLD = -100
CCI_OVERBOUGHT = 100

FEAR_GREED_FEAR = 30
FEAR_GREED_GREED = 70

# Bollinger Bands thresholds
BB_PERCENT_LOW = 0.2          # Near lower band → buy
BB_PERCENT_HIGH = 0.8         # Near upper band → sell

# MinMax Retracement thresholds
RETRACEMENT_LOW = 0.3
RETRACEMENT_HIGH = 0.7

# ADX trend strength threshold
ADX_TREND_THRESHOLD = 25

# MA Envelope threshold
MA_ENVELOPE_LOW = -1.0
MA_ENVELOPE_HIGH = 1.0

# Hurst exponent threshold
HURST_TRENDING = 0.5

# ─────────────────────────── Weight Optimization ───────────────────────────
WEIGHT_SMOOTHING_ALPHA = 0.3  # Exponential smoothing for accuracy
MIN_WEIGHT_FLOOR = 0.01       # Minimum 1% weight per indicator
SOFTMAX_TEMPERATURE = 1.0     # Softmax temperature for normalization

# ─────────────────────────── Ensemble & Win-Rate Optimization ───────────────────────────
ENSEMBLE_THRESHOLD = 0.5      # Weighted sum > this → BUY
# Confidence thresholds for neutral/hold zone
DEAD_ZONE_LOW = 0.35          # < 0.35 → SELL
DEAD_ZONE_HIGH = 0.65         # > 0.65 → BUY
ENABLE_DEAD_ZONE = True       # Hold position in between 0.35 and 0.65

# Weighting penalty
MIN_HISTORICAL_WIN_RATE = 0.50 # If win rate < 50% in train, weight = 0

# Stop-Loss / Take-Profit
ENABLE_SL_TP = True
STOP_LOSS_PCT = 0.05          # 5% stop loss
TAKE_PROFIT_PCT = 0.15        # 15% take profit

# ─────────────────────────── Backtesting ───────────────────────────
SELL_BEHAVIOR = "cash"        # "cash", "risk_free", "short"
INITIAL_CAPITAL = 10000.0

# ─────────────────────────── Volatility Adaptation ───────────────────────────
ENABLE_VOL_ADAPTIVE = True    # Adjust weights based on volatility regime
VOL_LOOKBACK = 12             # Months for volatility calculation
VOL_HIGH_THRESHOLD = 1.5      # Std devs above mean volatility → high-vol regime
OSCILLATOR_BOOST_FACTOR = 1.3 # Boost oscillator weights during high-vol
TREND_REDUCE_FACTOR = 0.7     # Reduce trend-following weights during high-vol

# ─────────────────────────── SHAP-Informed Initial Weights ───────────────────────────
# These indicators showed highest SHAP importance in the paper
HIGH_SHAP_INDICATORS = ["BB_MA", "VMAVG", "PTPS", "TMAVG"]
INITIAL_WEIGHT_BOOST = 1.5    # Boost factor for high-SHAP indicators (first iteration only)

# ─────────────────────────── Indicator Categories ───────────────────────────
INDICATOR_CATEGORIES = {
    "Trend": [
        "SMAVG", "EMAVG", "WMAVG", "TMAVG", "VMAVG",
        "BB_MA", "TRENDER_UP", "TRENDER_DN",
    ],
    "Momentum": [
        "MOMENTUM", "MOM_MA", "ROC", "MACD", "MACD_SIGNAL", "MACD_DIFF",
        "MAOsc", "MAO_SIGNAL", "MAO_DIFF",
    ],
    "Oscillator": [
        "RSI", "TAS_K", "TAS_D", "TAS_DS", "TAS_DSS",
        "WLPR", "CMCI", "FEAR_GREED",
    ],
    "Volatility": [
        "BB_UPPER", "BB_LOWER", "BB_WIDTH", "BB_PERCENT",
        "MAEs", "ADX", "ADXR", "DMI_PLUS", "DMI_MINUS",
        "HURST",
    ],
    "Support/Resistance": [
        "MIN", "MAX", "MM_RETRACEMENT", "PTPS",
    ],
}

# Flat set for quick lookup
OSCILLATOR_INDICATORS = set(INDICATOR_CATEGORIES["Oscillator"])
TREND_INDICATORS = set(INDICATOR_CATEGORIES["Trend"] + INDICATOR_CATEGORIES["Momentum"])
