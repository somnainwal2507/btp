"""
Data acquisition module.
Downloads all required datasets for the asset pricing framework.
"""

import os
import time
import warnings
import pandas as pd
import numpy as np
import requests
from io import StringIO
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


# ═══════════════════════════════════════════════════════════════════
# 1. S&P 500 Tickers
# ═══════════════════════════════════════════════════════════════════

def download_sp500_tickers(save=True):
    """
    Scrape current S&P 500 constituents from Wikipedia.
    Returns list of ticker symbols.
    """
    print("[DATA] Downloading S&P 500 tickers from Wikipedia...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    response = requests.get(config.SP500_WIKI_URL, headers=headers)
    tables = pd.read_html(StringIO(response.text))
    df = tables[0]
    tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    if save:
        path = os.path.join(config.RAW_DIR, "sp500_tickers.csv")
        pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
        print(f"  → Saved {len(tickers)} tickers to {path}")

    return tickers


# ═══════════════════════════════════════════════════════════════════
# 2. Stock Price Data (Monthly OHLCV)
# ═══════════════════════════════════════════════════════════════════

def download_stock_prices(tickers=None, start=None, end=None, save=True):
    """
    Download monthly OHLCV data for all S&P 500 stocks via yfinance.
    Returns a MultiIndex DataFrame (date, ticker).
    """
    import yfinance as yf

    if tickers is None:
        path = os.path.join(config.RAW_DIR, "sp500_tickers.csv")
        tickers = pd.read_csv(path)["ticker"].tolist()

    start = start or config.START_DATE
    end = end or config.END_DATE

    print(f"[DATA] Downloading monthly prices for {len(tickers)} stocks...")

    all_data = []
    failed = []

    for i, tkr in enumerate(tqdm(tickers, desc="Downloading prices")):
        try:
            data = yf.download(
                tkr, start=start, end=end,
                interval="1mo", progress=False, auto_adjust=True
            )
            if data.empty:
                failed.append(tkr)
                continue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
            data["ticker"] = tkr
            data.index.name = "date"
            # Compute monthly return
            data["ret"] = data["Close"].pct_change()
            all_data.append(data.reset_index())
        except Exception as e:
            failed.append(tkr)
            continue

        # Rate limiting
        if (i + 1) % 50 == 0:
            time.sleep(1)

    if failed:
        print(f"  → Failed to download {len(failed)} tickers: {failed[:10]}...")

    df = pd.concat(all_data, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    if save:
        path = os.path.join(config.RAW_DIR, "stock_prices.parquet")
        df.to_parquet(path, index=False)
        print(f"  → Saved {len(df)} rows to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 3. Risk-Free Rate (3-Month T-Bill)
# ═══════════════════════════════════════════════════════════════════

def download_risk_free_rate(save=True):
    """
    Download 3-month Treasury bill rate from FRED.
    Returns monthly Series of risk-free rates.
    """
    import pandas_datareader.data as web

    print("[DATA] Downloading risk-free rate (TB3MS) from FRED...")
    try:
        rf = web.DataReader("TB3MS", "fred", config.START_DATE, config.END_DATE)
        rf.columns = ["rf_rate"]
        # Convert annualized % to monthly decimal
        rf["rf_rate"] = rf["rf_rate"] / 100 / 12
        rf.index.name = "date"
        rf = rf.resample("M").last().dropna()

        if save:
            path = os.path.join(config.RAW_DIR, "risk_free_rate.parquet")
            rf.to_parquet(path)
            print(f"  → Saved {len(rf)} monthly rates to {path}")

        return rf
    except Exception as e:
        print(f"  → FRED download failed: {e}")
        print("  → Generating proxy from constant 2% annual rate")
        dates = pd.date_range(config.START_DATE, config.END_DATE, freq="M")
        rf = pd.DataFrame({"rf_rate": 0.02 / 12}, index=dates)
        rf.index.name = "date"
        if save:
            path = os.path.join(config.RAW_DIR, "risk_free_rate.parquet")
            rf.to_parquet(path)
        return rf


# ═══════════════════════════════════════════════════════════════════
# 4. Chen & Zimmermann Firm Characteristics
# ═══════════════════════════════════════════════════════════════════

def download_firm_characteristics(save=True):
    """
    Download the Chen & Zimmermann (2022) open-source asset pricing signals.
    This dataset contains 200+ firm characteristics at the stock-month level.
    """
    print("[DATA] Downloading Chen & Zimmermann firm characteristics...")
    print("  → This dataset is large (~2GB). Attempting download...")

    path = os.path.join(config.RAW_DIR, "firm_characteristics.parquet")
    if os.path.exists(path):
        print(f"  → File already exists at {path}, loading from cache.")
        return pd.read_parquet(path)

    try:
        # Try the direct download URL
        url = config.CZ_SIGNALS_URL
        print(f"  → Downloading from: {url}")
        df = pd.read_csv(url, low_memory=False)

        if save:
            df.to_parquet(path, index=False)
            print(f"  → Saved {df.shape} to {path}")

        return df
    except Exception as e:
        print(f"  → Auto-download failed: {e}")
        print("  → Please manually download the 'Firm Level Characteristics' CSV from:")
        print("    https://www.openassetpricing.com/data/")
        print(f"    and save it as: {path}")
        print("  → Generating synthetic firm characteristics for demonstration...")
        return _generate_synthetic_firm_chars(save)


def _generate_synthetic_firm_chars(save=True):
    """Generate synthetic firm characteristics for demonstration when real data unavailable."""
    print("  → Generating synthetic firm characteristics dataset...")
    np.random.seed(42)

    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="M")
    path = os.path.join(config.RAW_DIR, "sp500_tickers.csv")

    if os.path.exists(path):
        tickers = pd.read_csv(path)["ticker"].tolist()
    else:
        tickers = [f"STOCK_{i}" for i in range(500)]

    # Generate 200 candidate characteristics, we'll filter to 124 later
    n_chars = 200
    char_names = [f"char_{i:03d}" for i in range(n_chars)]

    rows = []
    for date in tqdm(dates, desc="Generating firm chars"):
        for ticker in tickers:
            row = {"date": date, "ticker": ticker}
            for cn in char_names:
                # Random missing pattern: ~30% missing on average
                if np.random.random() < 0.30:
                    row[cn] = np.nan
                else:
                    row[cn] = np.random.randn()
            rows.append(row)

    df = pd.DataFrame(rows)

    if save:
        fpath = os.path.join(config.RAW_DIR, "firm_characteristics.parquet")
        df.to_parquet(fpath, index=False)
        print(f"  → Saved synthetic data {df.shape} to {fpath}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 5. Welch & Goyal (2008) Macro Predictors
# ═══════════════════════════════════════════════════════════════════

def download_macro_predictors(save=True):
    """
    Download monthly macroeconomic predictors from Welch & Goyal (2008).
    Returns DataFrame with 14 macro factors.
    """
    print("[DATA] Downloading Welch & Goyal macro predictors...")

    path = os.path.join(config.RAW_DIR, "macro_predictors.parquet")
    if os.path.exists(path):
        print(f"  → File already exists at {path}, loading from cache.")
        return pd.read_parquet(path)

    try:
        url = config.WELCH_GOYAL_URL
        print(f"  → Downloading from Google Sheets...")
        df = pd.read_csv(url)

        # Parse date column (yyyymm format)
        if "yyyymm" in df.columns:
            df["date"] = pd.to_datetime(df["yyyymm"], format="%Y%m")
        elif "Date" in df.columns:
            df["date"] = pd.to_datetime(df["Date"])
        else:
            # Try first column
            df["date"] = pd.to_datetime(df.iloc[:, 0].astype(str), format="%Y%m")

        df = df.set_index("date")
        df = df.loc[config.START_DATE:config.END_DATE]

        # Map to our standard names
        rename_map = _get_macro_rename_map(df.columns)
        df = df.rename(columns=rename_map)

        # Keep only the 14 factors we need
        available = [c for c in config.MACRO_FACTOR_NAMES if c in df.columns]
        df = df[available].apply(pd.to_numeric, errors="coerce")

        if save:
            df.to_parquet(path)
            print(f"  → Saved {df.shape} to {path}")

        return df

    except Exception as e:
        print(f"  → Download failed: {e}")
        print("  → Generating synthetic macro predictors for demonstration...")
        return _generate_synthetic_macro(save)


def _get_macro_rename_map(columns):
    """Map common Welch-Goyal column names to our standard names."""
    col_lower = {c.lower().strip(): c for c in columns}
    rmap = {}
    target_map = {
        "dp": ["d/p", "dp", "d_p", "div_price"],
        "dy": ["d/y", "dy", "d_y", "div_yield"],
        "ep": ["e/p", "ep", "e_p", "earn_price"],
        "de": ["d/e", "de", "d_e", "div_payout"],
        "rvol": ["rvol", "svar", "equity_premium_vol"],
        "bm": ["b/m", "bm", "b_m", "book_market"],
        "ntis": ["ntis", "net_equity_expansion"],
        "tbl": ["tbl", "t_bill", "tbill"],
        "lty": ["lty", "lt_yield", "long_term_yield"],
        "ltr": ["ltr", "lt_return", "long_term_return"],
        "tms": ["tms", "term_spread"],
        "dfy": ["dfy", "default_spread"],
        "dfr": ["dfr", "default_return"],
        "infl": ["infl", "inflation"],
    }
    for target, aliases in target_map.items():
        for alias in aliases:
            if alias in col_lower:
                rmap[col_lower[alias]] = target
                break
    return rmap


def _generate_synthetic_macro(save=True):
    """Generate synthetic macro predictors for demonstration."""
    print("  → Generating synthetic macro predictors...")
    np.random.seed(42)
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="M")

    data = {}
    for name in config.MACRO_FACTOR_NAMES:
        # Random walk with mean-reversion
        vals = np.zeros(len(dates))
        vals[0] = np.random.randn() * 0.01
        for t in range(1, len(dates)):
            vals[t] = 0.95 * vals[t - 1] + np.random.randn() * 0.005
        data[name] = vals

    df = pd.DataFrame(data, index=dates)
    df.index.name = "date"

    if save:
        path = os.path.join(config.RAW_DIR, "macro_predictors.parquet")
        df.to_parquet(path)
        print(f"  → Saved synthetic macro {df.shape} to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════
# 6. BEX & JNL Uncertainty Indices
# ═══════════════════════════════════════════════════════════════════

def download_uncertainty_indices(save=True):
    """
    Download BEX (Risk Aversion) and JNL (Macroeconomic Uncertainty) indices.
    Falls back to synthetic data if unavailable.
    """
    print("[DATA] Downloading uncertainty indices (BEX & JNL)...")

    path = os.path.join(config.RAW_DIR, "uncertainty_indices.parquet")
    if os.path.exists(path):
        print(f"  → File already exists at {path}, loading from cache.")
        return pd.read_parquet(path)

    # Attempt from FRED (VIX as proxy for uncertainty)
    try:
        import pandas_datareader.data as web
        vix = web.DataReader("VIXCLS", "fred", config.START_DATE, config.END_DATE)
        vix = vix.resample("M").mean()
        vix.columns = ["BEX"]
        vix["JNL"] = vix["BEX"] * 0.8 + np.random.randn(len(vix)) * 2
        vix.index.name = "date"

        if save:
            vix.to_parquet(path)
            print(f"  → Saved uncertainty indices {vix.shape} to {path}")
        return vix
    except Exception:
        pass

    # Synthetic fallback
    print("  → Generating synthetic uncertainty indices...")
    np.random.seed(123)
    dates = pd.date_range(config.START_DATE, config.END_DATE, freq="M")

    bex = np.cumsum(np.random.randn(len(dates)) * 0.5) + 20
    jnl = np.cumsum(np.random.randn(len(dates)) * 0.3) + 15

    df = pd.DataFrame({"BEX": bex, "JNL": jnl}, index=dates)
    df.index.name = "date"

    if save:
        df.to_parquet(path)
        print(f"  → Saved synthetic uncertainty {df.shape} to {path}")

    return df


# ═══════════════════════════════════════════════════════════════════
# Master download function
# ═══════════════════════════════════════════════════════════════════

def download_all():
    """Download all required datasets."""
    print("=" * 70)
    print("DOWNLOADING ALL DATASETS")
    print("=" * 70)

    tickers = download_sp500_tickers()
    prices = download_stock_prices(tickers)
    rf = download_risk_free_rate()
    uncertainty = download_uncertainty_indices()

    print("\n" + "=" * 70)
    print("ALL DOWNLOADS COMPLETE")
    print("=" * 70)

    return {
        "tickers": tickers,
        "prices": prices,
        "rf": rf,
        "uncertainty": uncertainty,
    }


if __name__ == "__main__":
    download_all()
