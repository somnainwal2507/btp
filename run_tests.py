"""
run_tests.py — Automated Test Suite for core_engine.py
======================================================
Downloads SPY & TSLA data via yfinance in two timeframes,
runs each through the core engine, and exports Excel reports.

Usage:  python run_tests.py
"""

import os, sys, warnings, time
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    raise ImportError("Install yfinance:  pip install yfinance")

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core_engine import run_engine

# ─── Configuration ─────────────────────────────────────────────────
ASSETS = ["SPY", "TSLA"]
TIMEFRAMES = {
    "daily":   {"period": "10y",  "interval": "1d"},
    "monthly": {"period": "max",  "interval": "1mo"},
}
TEST_DIR  = os.path.join(os.path.dirname(__file__), "test_data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")


# ═══════════════════════════════════════════════════════════════════
#  1 · DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Download OHLCV data via yfinance and return a clean DataFrame.
    Handles timezone-aware indices, NaNs, and flat-line detection.
    """
    print(f"\n[TEST] Downloading {ticker}  period={period}  interval={interval}")
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval=interval, auto_adjust=True)

    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {ticker}")

    # Remove timezone info (pandas_ta can choke on tz-aware indices)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Keep standard OHLCV columns only
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    df = df[keep].copy()

    # Drop any fully-NaN rows
    df = df.dropna(how="all")

    # Forward-fill small gaps, then drop remaining NaN rows
    df = df.ffill().dropna()

    # Detect flat-line data (price stuck at single value)
    if df["Close"].nunique() < 3:
        print(f"  ⚠ WARNING: {ticker} has flat/constant price data")

    print(f"  → {len(df)} clean rows  |  "
          f"{df.index.min().date()} → {df.index.max().date()}")
    return df


def save_xlsx(df: pd.DataFrame, path: str):
    """Save DataFrame to .xlsx with a date column."""
    out = df.copy()
    out.index.name = "Date"
    out.to_excel(path)
    print(f"  → Saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  2 · TEST RUNNER
# ═══════════════════════════════════════════════════════════════════

def run_single_test(ticker: str, freq_label: str, file_path: str):
    """
    Run core_engine on one dataset and return results + metrics.
    Catches all exceptions so the suite never crashes.
    """
    tag = f"{ticker}_{freq_label}"
    print(f"\n{'═' * 60}")
    print(f"  TEST: {tag}")
    print(f"{'═' * 60}")

    try:
        result = run_engine(file_path)
        return {"tag": tag, "status": "PASS", "result": result}
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return {"tag": tag, "status": "FAIL", "error": str(e), "result": None}


def export_report(tag: str, result: dict, out_dir: str):
    """Write an Excel workbook with metrics + trade log."""
    if result is None:
        return
    path = os.path.join(out_dir, f"{tag}_test_results.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        # Sheet 1: comparative metrics
        result["comparison_df"].to_excel(w, sheet_name="Metrics")

        # Sheet 2: ensemble trade log (first 5000 rows max)
        trades = result["ensemble_trades"]
        if trades is not None and len(trades):
            trades.head(5000).to_excel(w, sheet_name="Ensemble Trades", index=False)

        # Sheet 3: per-indicator trade counts
        ind_summary = []
        for name, t in result.get("individual_trades", {}).items():
            if t is not None and len(t):
                ind_summary.append({
                    "Indicator": name,
                    "Trades": len(t),
                    "Win Rate": t["correct"].mean(),
                    "Cum Return": t["cum_strategy"].iloc[-1] - 1
                                  if "cum_strategy" in t.columns else 0,
                })
        if ind_summary:
            pd.DataFrame(ind_summary).to_excel(
                w, sheet_name="Individual Summary", index=False)

    print(f"  → Report saved: {path}")


# ═══════════════════════════════════════════════════════════════════
#  3 · MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = []
    t0 = time.time()

    for ticker in ASSETS:
        for freq_label, params in TIMEFRAMES.items():
            tag = f"{ticker}_{freq_label}"

            # Download & save
            try:
                df = download_data(ticker, params["period"], params["interval"])
            except Exception as e:
                print(f"  ✗ Download failed for {tag}: {e}")
                summary.append({"Test": tag, "Status": "DOWNLOAD_FAIL"})
                continue

            file_path = os.path.join(TEST_DIR, f"{tag}.xlsx")
            save_xlsx(df, file_path)

            # Run engine
            out = run_single_test(ticker, freq_label, file_path)
            summary.append({
                "Test": out["tag"],
                "Status": out["status"],
                "Ensemble Win Rate": (
                    out["result"]["comparison_df"].loc["Ensemble", "Win Rate"]
                    if out["result"] is not None
                       and "Ensemble" in out["result"]["comparison_df"].index
                    else None
                ),
            })

            # Export report
            if out["result"] is not None:
                export_report(tag, out["result"], RESULTS_DIR)

    elapsed = time.time() - t0

    # Print summary table
    print(f"\n\n{'═' * 60}")
    print("  TEST SUITE SUMMARY")
    print(f"{'═' * 60}")
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
