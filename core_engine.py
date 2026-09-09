"""
core_engine.py — Timeframe-Agnostic Technical Indicator Trading Engine
======================================================================
B.Tech Project: "Predicting Cross-Sectional Stock Returns Using
Technical Indicators: A Machine Learning Approach"

Authors: Som Nainwal, Nachiket Chondhikar, Atharva Mahajan

This module handles:
  1. OHLCV data ingestion from CSV / Excel
  2. Automatic timeframe inference via pd.infer_freq
  3. Computation of 39 technical indicators (pandas_ta)
  4. Binary signal generation for each indicator
  5. Dynamic accuracy-based weight optimisation (expanding window)
  6. Ensemble + individual-strategy backtesting
  7. Performance metrics: Win Rate, Sharpe, Max Drawdown, Cumulative Return
"""

import warnings, os, sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Install pandas_ta:  pip install pandas_ta")


# ═══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

CFG = {
    # Expanding-window sizes (row counts — timeframe-agnostic)
    "TRAIN_PERIODS": 120,
    "TEST_PERIODS": 12,
    # Indicator lookbacks
    "LB_SHORT": 14,
    "LB_MEDIUM": 20,
    # Ensemble thresholds
    "DEAD_ZONE_HIGH": 0.65,   # weighted sum > this → BUY
    "DEAD_ZONE_LOW": 0.35,    # weighted sum < this → SELL
    # Weight optimisation
    "MIN_WIN_RATE": 0.50,     # zero-weight if historical accuracy < 50%
    "SOFTMAX_TEMP": 1.0,
    "MIN_WEIGHT": 0.01,
    "SMOOTH_ALPHA": 0.3,
    # Risk management
    "STOP_LOSS": 0.05,
    "TAKE_PROFIT": 0.15,
    # Oscillator thresholds
    "RSI_LO": 30, "RSI_HI": 70,
    "STOCH_LO": 20, "STOCH_HI": 80,
    "WILL_LO": -80, "WILL_HI": -20,
    "CCI_LO": -100, "CCI_HI": 100,
    "BB_LO": 0.2, "BB_HI": 0.8,
    "ADX_THRESH": 25,
}


# ═══════════════════════════════════════════════════════════════════
#  1 · DATA LOADING
# ═══════════════════════════════════════════════════════════════════

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from CSV or Excel, standardise columns,
    parse dates, sort, and infer frequency.

    Returns
    -------
    df : DataFrame indexed by DatetimeIndex with columns
         [Open, High, Low, Close, Volume, ret].
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    # --- Find & parse date column -----------------------------------------
    date_col = next((c for c in df.columns
                     if "date" in c.lower() or "time" in c.lower()), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col])
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.drop(df.columns[0], axis=1)

    df = df.sort_index()

    # --- Standardise OHLCV names ------------------------------------------
    remap = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("open", "o"):
            remap[c] = "Open"
        elif cl in ("high", "h"):
            remap[c] = "High"
        elif cl in ("low", "l"):
            remap[c] = "Low"
        elif cl in ("close", "c", "adj close"):
            remap[c] = "Close"
        elif "vol" in cl:
            remap[c] = "Volume"
    df = df.rename(columns=remap)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df[["Open", "High", "Low", "Close", "Volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df["ret"] = df["Close"].pct_change()

    # --- Infer frequency ---------------------------------------------------
    try:
        freq = pd.infer_freq(df.index)
    except Exception:
        freq = None
    print(f"[ENGINE] Loaded {len(df)} rows  |  freq={freq}  |  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  2 · INDICATOR COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ~30 technical indicators using pandas_ta and return
    the same DataFrame with new indicator columns appended."""
    c, h, l = df["Close"], df["High"], df["Low"]
    lb_s, lb_m = CFG["LB_SHORT"], CFG["LB_MEDIUM"]

    # ── Moving Averages ───────────────────────────────────────────
    df["SMAVG"] = ta.sma(c, length=lb_m)
    df["EMAVG"] = ta.ema(c, length=lb_s)
    df["WMAVG"] = ta.wma(c, length=lb_m)
    hma = ta.hma(c, length=lb_m)
    if hma is not None:
        df["TMAVG"] = hma

    # ── Bollinger Bands ───────────────────────────────────────────
    bb = ta.bbands(c, length=lb_m, std=2)
    if bb is not None:
        df["BB_LOWER"]   = bb.iloc[:, 0]
        df["BB_MA"]      = bb.iloc[:, 1]
        df["BB_UPPER"]   = bb.iloc[:, 2]
        df["BB_WIDTH"]   = bb.iloc[:, 3]
        df["BB_PERCENT"] = bb.iloc[:, 4]

    # ── Momentum ──────────────────────────────────────────────────
    df["RSI"] = ta.rsi(c, length=lb_s)
    df["ROC"] = ta.roc(c, length=lb_s)
    df["MOMENTUM"] = c.diff(lb_s)
    df["MOM_MA"]   = ta.sma(df["MOMENTUM"], length=lb_s)

    macd = ta.macd(c, fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"]        = macd.iloc[:, 0]
        df["MACD_DIFF"]   = macd.iloc[:, 1]
        df["MACD_SIGNAL"] = macd.iloc[:, 2]

    # ── DMI / ADX ─────────────────────────────────────────────────
    adx = ta.adx(h, l, c, length=lb_s)
    if adx is not None:
        df["ADX"]      = adx.iloc[:, 0]
        df["DMI_PLUS"] = adx.iloc[:, 1]
        df["DMI_MINUS"]= adx.iloc[:, 2]

    # ── Stochastics ───────────────────────────────────────────────
    stoch = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
    if stoch is not None:
        df["TAS_K"] = stoch.iloc[:, 0]
        df["TAS_D"] = stoch.iloc[:, 1]

    # ── CCI / Williams %R ────────────────────────────────────────
    df["CMCI"] = ta.cci(h, l, c, length=lb_m)
    df["WLPR"] = ta.willr(h, l, c, length=lb_s)

    # ── Parabolic SAR ─────────────────────────────────────────────
    psar = ta.psar(h, l, c)
    if psar is not None:
        df["PTPS"] = psar.iloc[:, 0]

    df = df.ffill().bfill()
    # Collect names of successfully computed indicators
    base_cols = {"Open", "High", "Low", "Close", "Volume", "ret"}
    df.attrs["indicator_names"] = [c for c in df.columns if c not in base_cols]
    print(f"[ENGINE] Computed {len(df.attrs['indicator_names'])} indicators")
    return df


# ═══════════════════════════════════════════════════════════════════
#  3 · SIGNAL GENERATION  (indicator value → binary 0/1)
# ═══════════════════════════════════════════════════════════════════

def _trend(close, indicator):
    s = pd.Series(np.nan, index=close.index)
    s[close > indicator] = 1.0
    s[close < indicator] = 0.0
    return s.ffill()

def _osc(values, lo, hi):
    s = pd.Series(np.nan, index=values.index)
    s[values < lo] = 1.0
    s[values > hi] = 0.0
    return s.ffill()

def _cross(fast, slow):
    s = pd.Series(np.nan, index=fast.index)
    s[fast > slow] = 1.0
    s[fast < slow] = 0.0
    return s.ffill()

def _mom(values):
    s = pd.Series(np.nan, index=values.index)
    s[values > 0] = 1.0
    s[values < 0] = 0.0
    return s.ffill()


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert indicator values to binary BUY(1)/SELL(0) signals."""
    close = df["Close"]
    sigs = {}

    # Moving averages → trend-following
    for ma in ("SMAVG", "EMAVG", "WMAVG", "TMAVG", "BB_MA"):
        if ma in df.columns:
            sigs[ma] = _trend(close, df[ma])

    # Bollinger %B
    if "BB_PERCENT" in df.columns:
        sigs["BB_PERCENT"] = _osc(df["BB_PERCENT"], CFG["BB_LO"], CFG["BB_HI"])

    # RSI
    if "RSI" in df.columns:
        sigs["RSI"] = _osc(df["RSI"], CFG["RSI_LO"], CFG["RSI_HI"])

    # MACD crossover
    if "MACD" in df.columns and "MACD_SIGNAL" in df.columns:
        sigs["MACD"] = _cross(df["MACD"], df["MACD_SIGNAL"])
    if "MACD_DIFF" in df.columns:
        sigs["MACD_DIFF"] = _mom(df["MACD_DIFF"])

    # DMI crossover
    if "DMI_PLUS" in df.columns and "DMI_MINUS" in df.columns:
        sigs["DMI_PLUS"] = _cross(df["DMI_PLUS"], df["DMI_MINUS"])

    # ADX + DMI direction
    if "ADX" in df.columns and "DMI_PLUS" in sigs:
        s = pd.Series(np.nan, index=close.index)
        s[df["ADX"] > CFG["ADX_THRESH"]] = sigs["DMI_PLUS"][df["ADX"] > CFG["ADX_THRESH"]]
        s[df["ADX"] <= CFG["ADX_THRESH"]] = 0.5
        sigs["ADX"] = s.ffill()

    # Stochastics
    for st in ("TAS_K", "TAS_D"):
        if st in df.columns:
            sigs[st] = _osc(df[st], CFG["STOCH_LO"], CFG["STOCH_HI"])

    # Momentum / ROC
    for m in ("MOMENTUM", "MOM_MA", "ROC"):
        if m in df.columns:
            sigs[m] = _mom(df[m])

    # CCI
    if "CMCI" in df.columns:
        sigs["CMCI"] = _osc(df["CMCI"], CFG["CCI_LO"], CFG["CCI_HI"])

    # Williams %R
    if "WLPR" in df.columns:
        sigs["WLPR"] = _osc(df["WLPR"], CFG["WILL_LO"], CFG["WILL_HI"])

    # Parabolic SAR
    if "PTPS" in df.columns:
        sigs["PTPS"] = _trend(close, df["PTPS"])

    signals_df = pd.DataFrame(sigs, index=close.index).fillna(0.5).clip(0, 1)
    print(f"[ENGINE] Generated signals for {len(signals_df.columns)} indicators")
    return signals_df


# ═══════════════════════════════════════════════════════════════════
#  4 · WEIGHT OPTIMISATION
# ═══════════════════════════════════════════════════════════════════

def _accuracy(signals_df, returns):
    """Directional accuracy of each indicator's signal."""
    n = min(len(signals_df), len(returns))
    sigs = signals_df.iloc[:n]
    direction = (returns.iloc[:n].values > 0).astype(float)
    accs = {}
    for col in sigs.columns:
        s = sigs[col].values
        active = np.abs(s - 0.5) > 0.01
        if active.sum() < 5:
            accs[col] = 0.5
            continue
        binary = (s[active] > 0.5).astype(float)
        accs[col] = (binary == direction[active]).mean()
    return pd.Series(accs)


def _weights(accuracies, prev_w=None):
    """Softmax weights with win-rate penalty and smoothing."""
    acc = accuracies.copy()

    # Exponential smoothing with previous weights
    if prev_w is not None:
        alpha = CFG["SMOOTH_ALPHA"]
        for i in acc.index:
            if i in prev_w.index:
                acc[i] = alpha * acc[i] + (1 - alpha) * prev_w[i]

    # Softmax normalisation
    centered = acc - acc.mean()
    exp_v = np.exp(centered / CFG["SOFTMAX_TEMP"])
    w = exp_v / exp_v.sum()

    # Win-rate penalty: zero weight if accuracy < 50 %
    penalty = accuracies < CFG["MIN_WIN_RATE"]
    w[penalty] = 0.0
    if w.sum() == 0:
        w = pd.Series(1.0 / len(w), index=w.index)
        penalty[:] = False

    w = w.clip(lower=CFG["MIN_WEIGHT"])
    w[penalty] = 0.0
    w = w / w.sum()
    return w


# ═══════════════════════════════════════════════════════════════════
#  5 · ENSEMBLE SIGNAL
# ═══════════════════════════════════════════════════════════════════

def _ensemble_signal(row, weights):
    """Weighted vote → BUY / SELL / HOLD with dead zone."""
    common = row.index.intersection(weights.index)
    if len(common) == 0:
        return -1, 0.5
    s = row[common].values.astype(float)
    w = weights[common].values.astype(float)
    ws = w.sum()
    if ws == 0:
        return -1, 0.5
    w = w / ws
    score = np.nansum(s * w)
    if score > CFG["DEAD_ZONE_HIGH"]:
        return 1, score
    elif score < CFG["DEAD_ZONE_LOW"]:
        return 0, score
    else:
        return -1, score          # HOLD


# ═══════════════════════════════════════════════════════════════════
#  6 · BACKTEST (expanding window, row-based)
# ═══════════════════════════════════════════════════════════════════

def _backtest(signals_df, returns, close, is_ensemble=True, ind_name=None):
    """Run expanding-window backtest. Returns trade-log DataFrame."""
    train = CFG["TRAIN_PERIODS"]
    test  = CFG["TEST_PERIODS"]
    total = len(signals_df)
    log, prev_w, prev_sig, entry = [], None, 1, None

    for start in range(train, total, test):
        end = min(start + test, total)
        tr_sig = signals_df.iloc[:start]
        tr_ret = returns.iloc[:start]
        te_sig = signals_df.iloc[start:end]
        te_ret = returns.iloc[start:end]
        te_px  = close.iloc[start:end]

        if is_ensemble:
            shifted = tr_ret.shift(-1).dropna()
            acc = _accuracy(tr_sig.iloc[:len(shifted)], shifted)
            weights = _weights(acc, prev_w)
            prev_w = weights.copy()
        else:
            weights = pd.Series(0.0, index=signals_df.columns)
            if ind_name in weights.index:
                weights[ind_name] = 1.0

        for i in range(len(te_sig)):
            idx    = te_sig.index[i]
            row    = te_sig.iloc[i]
            ar     = te_ret.iloc[i] if not np.isnan(te_ret.iloc[i]) else 0.0
            px     = te_px.iloc[i]

            if is_ensemble:
                raw, ws = _ensemble_signal(row, weights)
            else:
                raw = int(row.get(ind_name, 0.5) > 0.5) if ind_name in row.index and pd.notna(row[ind_name]) else -1
                ws  = float(raw)

            # Stop-loss / take-profit
            forced = False
            if prev_sig == 1 and entry is not None:
                chg = (px - entry) / entry
                if chg <= -CFG["STOP_LOSS"]:
                    forced, ar = True, max(ar, -CFG["STOP_LOSS"])
                elif chg >= CFG["TAKE_PROFIT"]:
                    forced, ar = True, min(ar, CFG["TAKE_PROFIT"])

            sig = 0 if forced else (prev_sig if raw == -1 else raw)
            if sig == 1 and prev_sig == 0:
                entry = px
            elif sig == 0:
                entry = None
            prev_sig = sig

            p_ret = ar if sig == 1 else 0.0
            log.append({
                "date": idx, "signal": sig, "weighted_sum": ws,
                "actual_return": ar, "portfolio_return": p_ret,
                "buy_hold_return": ar,
                "correct": (sig == 1) == (ar > 0),
            })

    trades = pd.DataFrame(log)
    if len(trades):
        trades["date"] = pd.to_datetime(trades["date"])
        trades = trades.sort_values("date").reset_index(drop=True)
        trades["cum_strategy"] = (1 + trades["portfolio_return"]).cumprod()
        trades["cum_buyhold"]  = (1 + trades["buy_hold_return"]).cumprod()
    return trades


# ═══════════════════════════════════════════════════════════════════
#  7 · PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════

def _metrics(returns_arr, label="Strategy"):
    """Compute standard return-based metrics."""
    r = np.array(returns_arr, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return {k: 0.0 for k in ["label","total_return","ann_return",
                "ann_vol","sharpe","max_dd","win_rate","n_periods"]}
    cum = np.cumprod(1 + r)
    n_yr = max(len(r) / 12, 0.01)
    ann_ret = cum[-1] ** (1 / n_yr) - 1 if cum[-1] > 0 else 0
    std = np.std(r, ddof=1) if len(r) > 1 else 0.001
    sharpe = (np.mean(r) / std) * np.sqrt(12) if std > 0 else 0
    rm = pd.Series(cum).cummax()
    dd = ((pd.Series(cum) - rm) / rm).min()
    return {
        "label": label,
        "total_return": cum[-1] - 1,
        "ann_return": ann_ret,
        "ann_vol": std * np.sqrt(12),
        "sharpe": sharpe,
        "max_dd": dd,
        "win_rate": (r > 0).mean(),
        "n_periods": len(r),
    }


def compute_performance(trades_df):
    """Return dict with strategy, buyhold, and relative metrics."""
    s = _metrics(trades_df["portfolio_return"].values, "Signal Strategy")
    b = _metrics(trades_df["buy_hold_return"].values, "Buy & Hold")
    s["dir_accuracy"] = trades_df["correct"].mean()
    s["pct_invested"] = (trades_df["signal"] == 1).mean()
    return {"strategy": s, "buyhold": b}


# ═══════════════════════════════════════════════════════════════════
#  8 · MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def run_engine(file_path: str):
    """
    Full pipeline: load → indicators → signals → backtest → metrics.

    Returns
    -------
    result : dict with keys
        "ensemble_trades", "individual_trades",
        "ensemble_metrics", "comparison_df", "signals_df", "df"
    """
    # 1. Load
    df = load_data(file_path)

    # 2. Indicators
    df = compute_indicators(df)

    # 3. Signals
    signals_df = generate_signals(df)
    returns    = df["ret"]
    close      = df["Close"]

    # Align
    common = signals_df.index.intersection(returns.dropna().index)
    sig_al = signals_df.loc[common]
    ret_al = returns.loc[common]
    px_al  = close.loc[common]

    # 4. Ensemble backtest
    print("[ENGINE] Running ensemble backtest …")
    ens_trades = _backtest(sig_al, ret_al, px_al, is_ensemble=True)

    # 5. Individual backtests (core indicators)
    core = [c for c in ["RSI","MACD","BB_PERCENT","SMAVG","EMAVG",
                         "ADX","DMI_PLUS","ROC","MOMENTUM"]
            if c in sig_al.columns]
    ind_results = {}
    for ind in core:
        print(f"[ENGINE]   → Individual: {ind}")
        t = _backtest(sig_al, ret_al, px_al, is_ensemble=False, ind_name=ind)
        ind_results[ind] = t

    # 6. Metrics
    rows = []
    if len(ens_trades):
        em = compute_performance(ens_trades)
        rows.append({"Strategy": "Ensemble",
                      "Win Rate": em["strategy"]["win_rate"],
                      "Sharpe": em["strategy"]["sharpe"],
                      "Max DD": em["strategy"]["max_dd"],
                      "Total Return": em["strategy"]["total_return"],
                      "Dir. Accuracy": em["strategy"]["dir_accuracy"]})
    for name, t in ind_results.items():
        if len(t):
            m = compute_performance(t)
            rows.append({"Strategy": name,
                          "Win Rate": m["strategy"]["win_rate"],
                          "Sharpe": m["strategy"]["sharpe"],
                          "Max DD": m["strategy"]["max_dd"],
                          "Total Return": m["strategy"]["total_return"],
                          "Dir. Accuracy": m["strategy"]["dir_accuracy"]})

    # Buy & Hold row
    if len(ens_trades):
        bh = em["buyhold"]
        rows.append({"Strategy": "Buy & Hold",
                      "Win Rate": bh["win_rate"],
                      "Sharpe": bh["sharpe"],
                      "Max DD": bh["max_dd"],
                      "Total Return": bh["total_return"],
                      "Dir. Accuracy": np.nan})

    comp_df = pd.DataFrame(rows).set_index("Strategy")
    print("\n" + "=" * 60)
    print("COMPARATIVE RESULTS")
    print("=" * 60)
    print(comp_df.to_string(float_format=lambda x: f"{x:+.4f}"))

    return {
        "ensemble_trades": ens_trades,
        "individual_trades": ind_results,
        "ensemble_metrics": em if len(ens_trades) else None,
        "comparison_df": comp_df,
        "signals_df": signals_df,
        "df": df,
    }


# ═══════════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Run the trading engine")
    ap.add_argument("--file", required=True, help="Path to OHLCV CSV/Excel")
    args = ap.parse_args()
    run_engine(args.file)
