"""
Long-short decile portfolio analysis.
Sorts stocks into deciles by predicted return each month,
computes long-short (D10 − D1) returns, and plots cumulative returns.
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


def build_decile_portfolios(predictions_df, n_deciles=10):
    """
    Sort stocks into decile portfolios by predicted return each month.

    Parameters
    ----------
    predictions_df : DataFrame with columns [date, ticker, y_true, y_pred].
    n_deciles : int, number of portfolios (default 10).

    Returns
    -------
    decile_returns : DataFrame with columns [date, decile, mean_return, n_stocks].
    """
    df = predictions_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Assign decile within each month
    def assign_decile(group):
        group = group.copy()
        try:
            group["decile"] = pd.qcut(
                group["y_pred"], q=n_deciles, labels=False, duplicates="drop"
            ) + 1  # 1-indexed: D1 (lowest) to D10 (highest)
        except ValueError:
            # Fallback if too few unique values for qcut
            group["decile"] = pd.cut(
                group["y_pred"].rank(method="first"),
                bins=n_deciles, labels=False
            ) + 1
        return group

    df = df.groupby("date", group_keys=False).apply(assign_decile)

    # Compute equal-weighted mean realized return per decile per month
    decile_returns = (
        df.groupby(["date", "decile"])
        .agg(mean_return=("y_true", "mean"), n_stocks=("ticker", "count"))
        .reset_index()
    )

    return decile_returns


def compute_long_short_returns(decile_returns, n_deciles=10):
    """
    Compute long-short portfolio returns: D10 (highest predicted) − D1 (lowest predicted).

    Parameters
    ----------
    decile_returns : DataFrame from build_decile_portfolios().
    n_deciles : int, number of deciles.

    Returns
    -------
    ls_returns : DataFrame with columns [date, long_return, short_return, ls_return].
    """
    # D10 = highest predicted (long), D1 = lowest predicted (short)
    d_top = decile_returns[decile_returns["decile"] == n_deciles].set_index("date")["mean_return"]
    d_bottom = decile_returns[decile_returns["decile"] == 1].set_index("date")["mean_return"]

    ls = pd.DataFrame({
        "long_return": d_top,
        "short_return": d_bottom,
    }).dropna()

    ls["ls_return"] = ls["long_return"] - ls["short_return"]
    ls = ls.reset_index()
    ls = ls.sort_values("date")

    return ls


def plot_cumulative_ls_returns(all_ls_returns, save_path=None):
    """
    Plot cumulative long-short decile portfolio returns for all models.

    Parameters
    ----------
    all_ls_returns : dict of {model_name: ls_returns DataFrame}.
    save_path : str, path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    markers = ["o", "s", "D", "^", "v"]

    for i, (model_name, ls_df) in enumerate(sorted(all_ls_returns.items())):
        ls_df = ls_df.sort_values("date")
        cumulative = (1 + ls_df["ls_return"]).cumprod()

        color = colors[i % len(colors)]
        ax.plot(
            ls_df["date"], cumulative,
            label=model_name, color=color, linewidth=1.5,
            marker=markers[i % len(markers)], markersize=2, alpha=0.85,
        )

    ax.set_title("Cumulative Long-Short Decile Portfolio Returns (D10 − D1)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Return (Growth of $1)", fontsize=12)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Breakeven")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  → Saved plot to {save_path}")

    plt.close(fig)

def compute_summary_stats(ls_df, model_name):
    """
    Computes annualized Sharpe, Max Drawdown, and Win Rate.
    """
    returns = ls_df["ls_return"]
    
    # Annualized Sharpe (assuming monthly data)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
    
    # Max Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()
    
    # Win Rate (% of months with positive return)
    win_rate = (returns > 0).mean()
    
    return {
        "Model": model_name,
        "Annualized Sharpe": round(sharpe, 2),
        "Max Drawdown": round(max_dd, 4),
        "Monthly Win Rate": round(win_rate, 2),
        "Total Return": round(cumulative.iloc[-1] - 1, 4)
    }

def plot_model_comparison(stats_df, save_path=None):
    """
    Creates a bar chart comparing Sharpe Ratio and Max Drawdown across models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot Sharpe Ratio
    stats_df.plot(kind='bar', x='Model', y='Annualized Sharpe', ax=ax1, color='#2563eb')
    ax1.set_title("Annualized Sharpe Ratio (Higher is Better)")
    ax1.axhline(0, color='black', linewidth=0.8)
    
    # Plot Max Drawdown (we take absolute value for easier visual comparison)
    stats_df['Max DD Abs'] = stats_df['Max Drawdown'].abs()
    stats_df.plot(kind='bar', x='Model', y='Max DD Abs', ax=ax2, color='#dc2626')
    ax2.set_title("Max Drawdown Magnitude (Lower is Better)")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_risk_metrics(ls_df, risk_free_rate=0.0):
    """
    Calculate Sharpe Ratio and Maximum Drawdown for the L-S portfolio.
    
    Parameters
    ----------
    ls_df : DataFrame with 'ls_return' column.
    risk_free_rate : float, monthly risk-free rate (default 0.0).
    """
    returns = ls_df["ls_return"]
    
    # 1. Sharpe Ratio (Annualized)
    # Formula: (Mean Return - RF) / Std Dev * sqrt(12)
    mean_excess_return = returns.mean() - risk_free_rate
    std_return = returns.std()
    
    if std_return == 0:
        sharpe = 0
    else:
        sharpe = (mean_excess_return / std_return) * np.sqrt(12)

    # 2. Maximum Drawdown
    # We calculate the cumulative wealth index first
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    return sharpe, max_drawdown

def run_portfolio_analysis(predictions_dfs=None):
    """
    Run the full long-short decile portfolio analysis.

    Parameters
    ----------
    predictions_dfs : dict of {model_name: DataFrame}, optional.
        If None, loads from saved CSVs in results/.

    Returns
    -------
    all_ls_returns : dict of {model_name: long-short returns DataFrame}.
    all_decile_returns : dict of {model_name: decile returns DataFrame}.
    """
    print("\n" + "=" * 70)
    print("LONG-SHORT DECILE PORTFOLIO ANALYSIS")
    print("=" * 70)

    # Load predictions if not provided
    if predictions_dfs is None:
        predictions_dfs = {}
        for f in os.listdir(config.RESULTS_DIR):
            if f.startswith("predictions_") and f.endswith(".csv"):
                model_name = f.replace("predictions_", "").replace(".csv", "")
                path = os.path.join(config.RESULTS_DIR, f)
                predictions_dfs[model_name] = pd.read_csv(path)

    all_ls_returns = {}
    all_decile_rows = []

    for model_name, pred_df in predictions_dfs.items():
        print(f"\n  [{model_name}]")

        # Build decile portfolios
        decile_ret = build_decile_portfolios(pred_df)

        # Compute long-short returns
        ls_ret = compute_long_short_returns(decile_ret)

        # Summary stats
        mean_ls = ls_ret["ls_return"].mean()
        cumul = (1 + ls_ret["ls_return"]).prod() - 1
        print(f"    → Mean monthly L-S return: {mean_ls:+.4f}")
        print(f"    → Total cumulative L-S return: {cumul:+.2%}")
        print(f"    → Months: {len(ls_ret)}")

        all_ls_returns[model_name] = ls_ret

        # Tag decile returns with model name for saving
        decile_ret["model"] = model_name
        all_decile_rows.append(decile_ret)
    
    for model_name, pred_df in predictions_dfs.items():
        print(f"\n  [{model_name}]")

        decile_ret = build_decile_portfolios(pred_df)
        ls_ret = compute_long_short_returns(decile_ret)

        # New Metrics Calculation
        sharpe, max_dd = compute_risk_metrics(ls_ret)
        
        mean_ls = ls_ret["ls_return"].mean()
        cumul = (1 + ls_ret["ls_return"]).prod() - 1
        
        print(f"    → Mean monthly L-S return: {mean_ls:+.4f}")
        print(f"    → Total cumulative return: {cumul:+.2%}")
        print(f"    → Annualized Sharpe Ratio: {sharpe:.2f}")
        print(f"    → Maximum Drawdown:        {max_dd:.2%}")
    
    summary_list = []

    for model_name, pred_df in predictions_dfs.items():
        # 1. Get returns
        decile_ret = build_decile_portfolios(pred_df)
        ls_ret = compute_long_short_returns(decile_ret)
        
        # 2. Compute Stats
        stats = compute_summary_stats(ls_ret, model_name)
        summary_list.append(stats)

    # 3. Create Comparison Table
    comparison_df = pd.DataFrame(summary_list)
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(comparison_df.to_string(index=False))

    # 4. Plot Comparison
    plot_model_comparison(comparison_df, save_path="model_comparison.png")
    # Save decile returns
    if all_decile_rows:
        combined_decile = pd.concat(all_decile_rows, ignore_index=True)
        path = os.path.join(config.RESULTS_DIR, "decile_portfolio_returns.csv")
        combined_decile.to_csv(path, index=False)
        print(f"\n  → Saved decile returns to {path}")

    # Plot cumulative returns
    plot_path = os.path.join(config.RESULTS_DIR, "long_short_cumulative.png")
    plot_cumulative_ls_returns(all_ls_returns, save_path=plot_path)

    return all_ls_returns, combined_decile if all_decile_rows else None


if __name__ == "__main__":
    run_portfolio_analysis()
