"""
Uncertainty analysis.
Evaluate model performance during high-uncertainty periods using BEX and JNL indices.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from training.rolling_window import compute_r2_oos


def load_uncertainty_indices():
    """Load BEX and JNL indices."""
    path = os.path.join(config.RAW_DIR, "uncertainty_indices.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Uncertainty indices not found at {path}. Run data/download.py first."
        )
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df


def classify_uncertainty_regimes(uncertainty_df):
    """
    Split periods into high/low uncertainty using median split.

    Returns
    -------
    regimes : DataFrame with columns [BEX_regime, JNL_regime] (values: 'High'/'Low').
    """
    regimes = pd.DataFrame(index=uncertainty_df.index)

    for col in ["BEX", "JNL"]:
        if col in uncertainty_df.columns:
            median_val = uncertainty_df[col].median()
            regimes[f"{col}_regime"] = np.where(
                uncertainty_df[col] >= median_val, "High", "Low"
            )
        else:
            regimes[f"{col}_regime"] = "Unknown"

    return regimes


def run_uncertainty_analysis(predictions_dict, model_names=None):
    """
    Evaluate model performance during high vs low uncertainty periods.

    Parameters
    ----------
    predictions_dict : dict {model_name: DataFrame with [date, ticker, y_true, y_pred]}.
    model_names : list of model names (default: all).

    Returns
    -------
    results : DataFrame with R²_OOS per model per regime.
    """
    print("=" * 70)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 70)

    # Load uncertainty indices
    uncertainty_df = load_uncertainty_indices()
    regimes = classify_uncertainty_regimes(uncertainty_df)

    if model_names is None:
        model_names = list(predictions_dict.keys())

    results = []

    for model_name in model_names:
        if model_name not in predictions_dict:
            continue

        pred_df = predictions_dict[model_name].copy()
        pred_df["date"] = pd.to_datetime(pred_df["date"])
        pred_df["month_end"] = pred_df["date"].dt.to_period("M").dt.to_timestamp("M")

        # Merge with regimes
        regimes_reset = regimes.copy()
        regimes_reset["month_end"] = regimes_reset.index.to_period("M").to_timestamp("M")

        merged = pred_df.merge(regimes_reset, on="month_end", how="left")

        for index_name in ["BEX", "JNL"]:
            regime_col = f"{index_name}_regime"
            if regime_col not in merged.columns:
                continue

            for regime in ["High", "Low"]:
                mask = merged[regime_col] == regime
                if mask.sum() < 10:
                    continue

                y_true = merged.loc[mask, "y_true"].values
                y_pred = merged.loc[mask, "y_pred"].values
                r2 = compute_r2_oos(y_true, y_pred)

                results.append({
                    "model": model_name,
                    "index": index_name,
                    "regime": regime,
                    "R2_OOS": r2,
                    "n_obs": mask.sum(),
                })

        # Overall for reference
        y_true_all = pred_df["y_true"].values
        y_pred_all = pred_df["y_pred"].values
        results.append({
            "model": model_name,
            "index": "Overall",
            "regime": "All",
            "R2_OOS": compute_r2_oos(y_true_all, y_pred_all),
            "n_obs": len(pred_df),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        print("\n[UNCERTAINTY] Results:")
        pivot = results_df.pivot_table(
            values="R2_OOS",
            index="model",
            columns=["index", "regime"],
        )
        print(pivot.round(4).to_string())

        # Save
        path = os.path.join(config.RESULTS_DIR, "uncertainty_analysis.csv")
        results_df.to_csv(path, index=False)
        print(f"\n  → Saved to {path}")

        # Plot
        _plot_uncertainty(results_df)

    return results_df


def _plot_uncertainty(results_df):
    """Plot R²_OOS across uncertainty regimes."""
    for index_name in ["BEX", "JNL"]:
        data = results_df[results_df["index"] == index_name]
        if len(data) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))

        models = data["model"].unique()
        x = np.arange(len(models))
        width = 0.35

        high_vals = data[data["regime"] == "High"].set_index("model")["R2_OOS"]
        low_vals = data[data["regime"] == "Low"].set_index("model")["R2_OOS"]

        ax.bar(x - width / 2, [high_vals.get(m, 0) for m in models],
               width, label="High Uncertainty", color="tomato", alpha=0.8)
        ax.bar(x + width / 2, [low_vals.get(m, 0) for m in models],
               width, label="Low Uncertainty", color="steelblue", alpha=0.8)

        ax.set_ylabel("R²_OOS", fontsize=12)
        ax.set_title(f"Model Performance by {index_name} Uncertainty Regime", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.axhline(0, color="black", linewidth=0.5)
        plt.tight_layout()

        path = os.path.join(config.RESULTS_DIR, f"uncertainty_{index_name}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → Saved {index_name} plot to {path}")
