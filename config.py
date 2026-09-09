"""
Configuration for the Cross-Sectional Asset Pricing Framework.
Uddin, Akter, and Hassan (2026) replication.
"""

import os

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Ensure directories exist
for d in [RAW_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────── Date Range ───────────────────────────
START_DATE = "1990-01-01"
END_DATE = "2021-12-31"
FIRST_TEST_YEAR = 2006
LAST_TEST_YEAR = 2021
TRAIN_START_YEAR = 1990
WINDOW_SHIFT_MONTHS = 12  # Expanding window shifts by 12 months

# ─────────────────────────── Feature Engineering ───────────────────────────
MAX_MISSING_RATIO = 0.60           # Exclude features with >60% missing

# ─────────────────────────── Technical Indicator Names ───────────────────────────
TECHNICAL_INDICATORS = [
    "BB_MA", "BB_UPPER", "BB_LOWER", "BB_WIDTH", "BB_PERCENT",
    "DMI_PLUS", "DMI_MINUS", "ADX", "ADXR",
    "EMAVG", "CMCI", "FEAR_GREED", "HURST",
    "MIN", "MAX", "MM_RETRACEMENT",
    "MOMENTUM", "MOM_MA",
    "MACD", "MACD_SIGNAL", "MACD_DIFF",
    "MAEs", "MAOsc", "MAO_SIGNAL", "MAO_DIFF",
    "ROC", "PTPS", "RSI", "SMAVG",
    "TAS_K", "TAS_D", "TAS_DS", "TAS_DSS",
    "TRENDER_UP", "TRENDER_DN",
    "TMAVG", "VMAVG", "WMAVG", "WLPR"
]

# ─────────────────────────── Model Hyperparameters ───────────────────────────
# LASSO
LASSO_ALPHA = 1e-3

# Decision Tree
DT_PARAMS = {
    "random_state": 42,
}

# Random Forest
RF_PARAMS = {
    "n_estimators": 50,
    "max_depth": 8,
    "random_state": 42,
    "n_jobs": -1,
}


# Deep Neural Network
DNN_PARAMS = {
    "hidden_layers": [64, 32, 16, 8, 4],
    "activation": "relu",
    "output_activation": "linear",
    "batch_size": 128,
    "learning_rate": 1e-3,
    "max_epochs": 200,
    "patience": 10,  # Early stopping patience
}

# ─────────────────────────── Data Source URLs ───────────────────────────
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

