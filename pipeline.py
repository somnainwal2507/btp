"""
Preprocessing pipeline.
Feature exclusion, forward fill, and cross-sectional mean imputation.
"""

import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def exclude_features(df, feature_cols, threshold=None):
    """
    Remove features with >threshold proportion of missing values.

    Parameters
    ----------
    df : DataFrame containing the features.
    feature_cols : list of feature column names to check.
    threshold : float, maximum allowed missing ratio (default from config).

    Returns
    -------
    df : DataFrame with excluded columns removed.
    kept_cols : list of retained feature column names.
    """
    threshold = threshold or config.MAX_MISSING_RATIO

    missing_ratios = df[feature_cols].isna().mean()
    excluded = missing_ratios[missing_ratios > threshold].index.tolist()
    kept_cols = [c for c in feature_cols if c not in excluded]

    print(f"[PREPROCESS] Feature exclusion (>{threshold*100:.0f}% missing):")
    print(f"  → Checked: {len(feature_cols)} features")
    print(f"  → Excluded: {len(excluded)} features")
    print(f"  → Kept: {len(kept_cols)} features")

    return df.drop(columns=excluded, errors="ignore"), kept_cols


def forward_fill(df, feature_cols):
    """
    Apply forward fill per stock for each feature.

    Parameters
    ----------
    df : DataFrame with [ticker] column and feature columns.
    feature_cols : list of feature column names.

    Returns
    -------
    df : DataFrame with forward-filled values.
    """
    print("[PREPROCESS] Applying forward fill per stock...")

    before_missing = df[feature_cols].isna().sum().sum()

    df = df.sort_values(["ticker", "date"])
    df[feature_cols] = df.groupby("ticker")[feature_cols].ffill()

    after_missing = df[feature_cols].isna().sum().sum()
    filled = before_missing - after_missing
    print(f"  → Filled {filled:,} values ({filled / max(before_missing, 1) * 100:.1f}%)")

    return df


def cross_sectional_mean_impute(df, feature_cols):
    """
    Replace remaining missing values with cross-sectional mean
    (mean of that feature for that specific month across all firms).

    Parameters
    ----------
    df : DataFrame with [date] column and feature columns.
    feature_cols : list of feature column names.

    Returns
    -------
    df : DataFrame with imputed values.
    """
    print("[PREPROCESS] Applying cross-sectional mean imputation...")

    before_missing = df[feature_cols].isna().sum().sum()

    # Group by month, fill with monthly cross-sectional mean
    df["_month"] = pd.to_datetime(df["date"]).dt.to_period("M")

    for col in feature_cols:
        monthly_mean = df.groupby("_month")[col].transform("mean")
        df[col] = df[col].fillna(monthly_mean)

    # Fill any remaining with global mean
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].mean())

    # Final fallback: fill with 0
    df[feature_cols] = df[feature_cols].fillna(0)

    df = df.drop(columns=["_month"])

    after_missing = df[feature_cols].isna().sum().sum()
    filled = before_missing - after_missing
    print(f"  → Imputed {filled:,} remaining values")
    print(f"  → Remaining missing: {after_missing:,}")

    return df


def preprocess(df, ti_names):
    """
    Full preprocessing pipeline:
    1. Exclude features with >60% missing.
    2. Forward fill per stock.
    3. Cross-sectional mean imputation.

    Parameters
    ----------
    df : DataFrame with all features.
    ti_names : list of 39 TI column names.

    Returns
    -------
    df : Preprocessed DataFrame.
    final_ti : list of remaining TI names after exclusion.
    """
    print("=" * 70)
    print("PREPROCESSING PIPELINE")
    print("=" * 70)

    # Step 1: Exclude features with too many missing values
    df, kept_features = exclude_features(df, ti_names)

    final_ti = [c for c in ti_names if c in kept_features]

    # Step 2: Forward fill per stock
    df = forward_fill(df, final_ti)

    # Step 3: Cross-sectional mean imputation
    df = cross_sectional_mean_impute(df, final_ti)

    # Drop rows with missing target
    before = len(df)
    df = df.dropna(subset=["target"])
    print(f"\n[PREPROCESS] Dropped {before - len(df)} rows with missing target")

    print(f"\n[PREPROCESS] Final dataset:")
    print(f"  → Shape: {df.shape}")
    print(f"  → Technical indicators: {len(final_ti)}")
    print(f"  → Date range: {df['date'].min()} to {df['date'].max()}")

    return df, final_ti
