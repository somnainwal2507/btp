"""
Macroeconomic predictors module.
PCA on Welch & Goyal (2008) factors + interaction terms.
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def load_macro_predictors():
    """Load macro predictors from disk."""
    path = os.path.join(config.RAW_DIR, "macro_predictors.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Macro predictors not found at {path}. Run data/download.py first."
        )
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    return df


def compute_pca(macro_df, n_components=None):
    """
    Perform PCA on the 14 macro factors.

    Parameters
    ----------
    macro_df : DataFrame with 14 macro columns.
    n_components : int, number of PCs to retain (default from config).

    Returns
    -------
    pca_df : DataFrame with principal component columns (PC_1, PC_2, PC_3).
    pca_model : fitted PCA object.
    scaler : fitted StandardScaler.
    """
    n_components = n_components or config.N_PCA_COMPONENTS

    print(f"[FEATURES] Computing PCA on {macro_df.shape[1]} macro factors...")

    # Standardize before PCA
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df.fillna(macro_df.mean()))

    pca_model = PCA(n_components=n_components)
    pcs = pca_model.fit_transform(macro_scaled)

    pc_names = [f"PC_{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pcs, index=macro_df.index, columns=pc_names)

    explained = pca_model.explained_variance_ratio_
    print(f"  → Explained variance: {explained}")
    print(f"  → Cumulative: {explained.cumsum()[-1]*100:.1f}%")

    return pca_df, pca_model, scaler


def create_interactions(firm_chars_df, char_names, pca_df):
    """
    Create interaction terms: each firm characteristic × (3 PCs + constant).

    Parameters
    ----------
    firm_chars_df : DataFrame with firm characteristics (indexed by date, ticker).
    char_names : list of characteristic column names.
    pca_df : DataFrame with PCA components (indexed by date).

    Returns
    -------
    interactions_df : DataFrame with 124 × 4 = 496 interaction columns.
    interaction_names : list of interaction column names.
    """
    print(f"[FEATURES] Creating interaction terms: "
          f"{len(char_names)} chars × (3 PCs + 1 const) = "
          f"{len(char_names) * 4} terms...")

    # Ensure firm_chars has a date column for merging
    if "date" in firm_chars_df.columns:
        dates = firm_chars_df["date"]
    else:
        dates = firm_chars_df.index.get_level_values("date")

    interaction_names = []
    interaction_data = {}

    for char in char_names:
        char_vals = firm_chars_df[char].values

        # Interaction with constant (just the characteristic itself)
        col_name = f"{char}_const"
        interaction_data[col_name] = char_vals
        interaction_names.append(col_name)

        # Interaction with each PC
        for pc_col in pca_df.columns:
            # Map PC values to each stock-month by date
            date_arr = pd.to_datetime(dates).values
            pc_map = pca_df[pc_col].to_dict()

            # Match by month-end
            pc_vals = np.array([
                pc_map.get(d, np.nan)
                for d in pd.to_datetime(date_arr).to_period("M").to_timestamp("M")
            ])

            col_name = f"{char}_{pc_col}"
            interaction_data[col_name] = char_vals * pc_vals
            interaction_names.append(col_name)

    interactions_df = pd.DataFrame(interaction_data, index=firm_chars_df.index)

    print(f"  → Created {len(interaction_names)} interaction terms")
    return interactions_df, interaction_names


def build_macro_features(firm_chars_df, char_names):
    """
    Full macro feature pipeline:
    1. Load macro predictors
    2. PCA → 3 components
    3. Interactions: 124 × 4 = 496
    4. Add 14 raw macro factors
    Total: 510 covariates

    Returns
    -------
    full_features : DataFrame with 510 covariate columns.
    feature_names : list of all 510 feature names.
    """
    # Load macro data
    macro_df = load_macro_predictors()
    available_macros = [c for c in config.MACRO_FACTOR_NAMES if c in macro_df.columns]

    # PCA
    pca_df, pca_model, scaler = compute_pca(macro_df[available_macros])

    # Interactions
    interactions_df, interaction_names = create_interactions(
        firm_chars_df, char_names, pca_df
    )

    # Add raw macro factors to each stock-month row
    if "date" in firm_chars_df.columns:
        dates = firm_chars_df["date"]
    else:
        dates = firm_chars_df.index.get_level_values("date")

    macro_expanded = {}
    for macro_col in available_macros:
        macro_map = macro_df[macro_col].to_dict()
        date_arr = pd.to_datetime(dates).values
        macro_expanded[macro_col] = np.array([
            macro_map.get(d, np.nan)
            for d in pd.to_datetime(date_arr).to_period("M").to_timestamp("M")
        ])

    macro_expanded_df = pd.DataFrame(macro_expanded, index=firm_chars_df.index)

    # Combine: interactions (496) + raw macro (14) = 510
    full_features = pd.concat([interactions_df, macro_expanded_df], axis=1)
    feature_names = interaction_names + available_macros

    print(f"[FEATURES] Total covariates: {len(feature_names)} "
          f"(target: {config.N_TOTAL_COVARIATES})")

    return full_features, feature_names, pca_model, scaler
