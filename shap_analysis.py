"""
SHAP analysis on the Deep-NN model.
Computes marginal contribution of the 39 technical indicators.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def run_shap_analysis(deep_nn_model, X_test, feature_names, ti_names,
                      n_background=100, n_explain=500):
    """
    Run SHAP analysis on the Deep-NN to evaluate technical indicators.

    Parameters
    ----------
    deep_nn_model : trained DeepNNModel instance.
    X_test : numpy array of test features.
    feature_names : list of all feature column names.
    ti_names : list of 39 technical indicator names.
    n_background : int, samples for background dataset.
    n_explain : int, samples to explain.

    Returns
    -------
    shap_importance : DataFrame with mean |SHAP| for each TI.
    """
    import shap
    import torch

    print("=" * 70)
    print("SHAP ANALYSIS ON DEEP-NN")
    print("=" * 70)

    # Scale test data
    X_scaled = deep_nn_model.scaler.transform(X_test)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Subsample for efficiency
    n_bg = min(n_background, len(X_scaled))
    n_exp = min(n_explain, len(X_scaled))

    bg_indices = np.random.choice(len(X_scaled), n_bg, replace=False)
    exp_indices = np.random.choice(len(X_scaled), n_exp, replace=False)

    background = torch.FloatTensor(X_scaled[bg_indices]).to(deep_nn_model.device)
    explain_data = torch.FloatTensor(X_scaled[exp_indices]).to(deep_nn_model.device)

    # SHAP DeepExplainer
    print(f"[SHAP] Computing SHAP values ({n_exp} samples, {n_bg} background)...")
    try:
        explainer = shap.DeepExplainer(deep_nn_model.get_torch_model(), background)
        shap_values = explainer.shap_values(explain_data)
    except Exception as e:
        print(f"  → DeepExplainer failed ({e}), falling back to KernelExplainer...")
        # Fallback to KernelExplainer
        def predict_fn(x):
            x_t = torch.FloatTensor(x).to(deep_nn_model.device)
            with torch.no_grad():
                return deep_nn_model.get_torch_model()(x_t).cpu().numpy()

        explainer = shap.KernelExplainer(predict_fn, X_scaled[bg_indices])
        shap_values = explainer.shap_values(X_scaled[exp_indices])

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if isinstance(shap_values, torch.Tensor):
        shap_values = shap_values.cpu().numpy()

    # Extract SHAP values for technical indicators
    ti_indices = [feature_names.index(ti) for ti in ti_names if ti in feature_names]
    ti_shap = shap_values[:, ti_indices] if len(ti_indices) > 0 else np.zeros((n_exp, 0))

    # Compute mean absolute SHAP value per TI
    ti_present = [ti for ti in ti_names if ti in feature_names]
    mean_abs_shap = np.abs(ti_shap).mean(axis=0)

    shap_importance = pd.DataFrame({
        "indicator": ti_present,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    print(f"\n[SHAP] Top 10 Technical Indicators by mean |SHAP|:")
    print(shap_importance.head(10).to_string(index=False))

    # Plot
    _plot_shap_bar(shap_importance)
    _plot_shap_summary(shap_values, feature_names, ti_indices, ti_present)

    # Save
    path = os.path.join(config.RESULTS_DIR, "shap_importance.csv")
    shap_importance.to_csv(path, index=False)
    print(f"\n  → Saved SHAP importance to {path}")

    return shap_importance


def _plot_shap_bar(shap_importance):
    """Plot horizontal bar chart of mean |SHAP| values."""
    fig, ax = plt.subplots(figsize=(10, 12))
    data = shap_importance.sort_values("mean_abs_shap", ascending=True)

    ax.barh(data["indicator"], data["mean_abs_shap"], color="steelblue")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Technical Indicator Importance (Deep-NN)", fontsize=14)
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, "shap_bar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved SHAP bar chart to {path}")


def _plot_shap_summary(shap_values, feature_names, ti_indices, ti_names):
    """Plot SHAP summary (beeswarm-style) for tech indicators."""
    if len(ti_indices) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 12))
    ti_shap = shap_values[:, ti_indices]

    # Simple violin-style summary
    parts = ax.violinplot(
        [ti_shap[:, i] for i in range(len(ti_names))],
        positions=range(len(ti_names)),
        vert=False,
        showmeans=True,
    )
    ax.set_yticks(range(len(ti_names)))
    ax.set_yticklabels(ti_names, fontsize=8)
    ax.set_xlabel("SHAP Value", fontsize=12)
    ax.set_title("SHAP Value Distribution — Technical Indicators", fontsize=14)
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Saved SHAP summary to {path}")
