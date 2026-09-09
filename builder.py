"""
Feature matrix builder.
Assembles the 39 technical indicators + target variable (excess returns).
"""

import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from features.technical import compute_technical_indicators_panel


def compute_excess_returns(prices_df, rf_df):
    """
    Compute monthly excess returns: r_{i,t+1} = stock_return - rf_rate.

    Parameters
    ----------
    prices_df : DataFrame with [date, ticker, ret] columns.
    rf_df : DataFrame indexed by date with [rf_rate] column.

    Returns
    -------
    DataFrame with [date, ticker, excess_ret] columns.
    """
    print("[FEATURES] Computing excess returns...")

    # Ensure date alignment
    prices = prices_df[["date", "ticker", "ret"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])

    rf = rf_df.copy()
    if "date" in rf.columns:
        rf = rf.set_index("date")
    rf.index = pd.to_datetime(rf.index)

    # Align by month-end
    prices["month_end"] = prices["date"].dt.to_period("M").dt.to_timestamp("M")
    rf["month_end"] = rf.index.to_period("M").to_timestamp("M")
    rf_map = rf.set_index("month_end")["rf_rate"].to_dict()

    prices["rf_rate"] = prices["month_end"].map(rf_map)
    prices["excess_ret"] = prices["ret"] - prices["rf_rate"]

    # Shift target: we predict NEXT month's excess return
    prices = prices.sort_values(["ticker", "date"])
    prices["target"] = prices.groupby("ticker")["excess_ret"].shift(-1)

    print(f"  → Computed excess returns for {prices['ticker'].nunique()} stocks")

    return prices[["date", "ticker", "ret", "rf_rate", "excess_ret", "target"]]


def build_feature_matrix():
    """
    Assemble the feature matrix:
    - 39 technical indicators
    - Target: next-month excess return

    Returns
    -------
    feature_matrix : DataFrame with all features and target.
    ti_names : list of 39 technical indicator names.
    """
    print("=" * 70)
    print("BUILDING FEATURE MATRIX")
    print("=" * 70)

    # 1. Load prices and RF rate
    prices_path = os.path.join(config.RAW_DIR, "stock_prices.parquet")
    rf_path = os.path.join(config.RAW_DIR, "risk_free_rate.parquet")

    prices_df = pd.read_parquet(prices_path)
    rf_df = pd.read_parquet(rf_path)

    # 2. Compute excess returns and target
    returns_df = compute_excess_returns(prices_df, rf_df)

    # 3. Compute technical indicators
    ti_df = compute_technical_indicators_panel(prices_df)

    # 4. Merge returns with technical indicators
    print("[FEATURES] Merging all features...")

    ti_cols = [c for c in config.TECHNICAL_INDICATORS if c in ti_df.columns]

    if "date" in ti_df.columns:
        ti_df["date"] = pd.to_datetime(ti_df["date"])

    merged = returns_df.merge(
        ti_df[["date", "ticker"] + ti_cols],
        on=["date", "ticker"],
        how="left",
    )

    # Validate
    print(f"\n[FEATURES] Feature matrix summary:")
    print(f"  → Shape: {merged.shape}")
    print(f"  → Technical indicators: {len(ti_cols)}")
    print(f"  → Date range: {merged['date'].min()} to {merged['date'].max()}")
    print(f"  → Stocks: {merged['ticker'].nunique()}")

    # Save
    output_path = os.path.join(config.RAW_DIR, "feature_matrix.parquet")
    merged.to_parquet(output_path, index=False)
    print(f"  → Saved to {output_path}")

    return merged, ti_cols


if __name__ == "__main__":
    build_feature_matrix()
