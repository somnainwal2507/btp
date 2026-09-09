"""
Firm characteristics processing.
Filters and aligns Chen & Zimmermann (2022) characteristics.
"""

import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_firm_characteristics():
    """Load raw firm characteristics from disk."""
    path = os.path.join(config.RAW_DIR, "firm_characteristics.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Firm characteristics not found at {path}. Run data/download.py first."
        )
    return pd.read_parquet(path)


def filter_characteristics(df, min_coverage=None):
    """
    Filter firm characteristics to keep only those with sufficient coverage.

    Parameters
    ----------
    df : DataFrame with columns including firm characteristics.
    min_coverage : float, minimum non-missing ratio (default from config).

    Returns
    -------
    DataFrame with filtered characteristics + identifier columns.
    """
    min_coverage = min_coverage or config.MIN_NONMISSING_RATIO

    # Identify characteristic columns (exclude date, ticker, identifiers)
    id_cols = ["date", "ticker", "permno", "PERMNO", "gvkey"]
    id_cols_present = [c for c in id_cols if c in df.columns]
    char_cols = [c for c in df.columns if c not in id_cols_present]

    # Compute coverage per characteristic
    coverage = df[char_cols].notna().mean()
    valid_chars = coverage[coverage >= min_coverage].index.tolist()

    print(f"[FEATURES] Firm characteristics: {len(char_cols)} total, "
          f"{len(valid_chars)} with ≥{min_coverage*100:.0f}% coverage")

    # Keep up to N_FIRM_CHARS characteristics with highest coverage
    if len(valid_chars) > config.N_FIRM_CHARS:
        coverage_sorted = coverage[valid_chars].sort_values(ascending=False)
        valid_chars = coverage_sorted.head(config.N_FIRM_CHARS).index.tolist()
        print(f"  → Trimmed to top {config.N_FIRM_CHARS} by coverage")

    result = df[id_cols_present + valid_chars].copy()
    return result, valid_chars


def process_firm_characteristics():
    """
    Full pipeline: load → filter → align → return panel DataFrame.

    Returns
    -------
    df : DataFrame indexed by (date, ticker) with 124 firm characteristics.
    char_names : list of characteristic column names.
    """
    print("[FEATURES] Processing firm characteristics...")

    df = load_firm_characteristics()

    # Ensure date is datetime
    if "date" not in df.columns:
        # Try common alternatives
        for col in ["Date", "DATE", "yyyymm"]:
            if col in df.columns:
                df = df.rename(columns={col: "date"})
                break

    df["date"] = pd.to_datetime(df["date"])

    # Ensure ticker column exists
    if "ticker" not in df.columns:
        for col in ["TICKER", "Ticker", "permno", "PERMNO"]:
            if col in df.columns:
                df = df.rename(columns={col: "ticker"})
                break

    # Filter to date range
    df = df[(df["date"] >= config.START_DATE) & (df["date"] <= config.END_DATE)]

    # Filter characteristics by coverage
    df, char_names = filter_characteristics(df)

    # Ensure we have exactly N_FIRM_CHARS or as many as available
    actual = len(char_names)
    if actual < config.N_FIRM_CHARS:
        print(f"  → Warning: Only {actual} characteristics available "
              f"(target: {config.N_FIRM_CHARS})")

    print(f"  → Final firm characteristics: {actual} features, "
          f"{df.shape[0]} stock-months")

    return df, char_names
