"""
Comprehensive visualization module for the signal-based trading system.
Generates publication-quality charts for weight evolution, cumulative returns,
signal distribution, indicator category contribution, and more.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_system import signal_config as sc

# ─── Style Configuration ───
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333355",
    "axes.labelcolor": "#e0e0e0",
    "text.color": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.facecolor": "#1a1a2e",
    "legend.edgecolor": "#333355",
})

# Color palette
COLORS = {
    "strategy": "#00d4ff",
    "buyhold": "#ff6b6b",
    "buy_signal": "#00e676",
    "sell_signal": "#ff5252",
    "neutral": "#ffd740",
    "grid": "#2a2a4a",
    "accent1": "#7c4dff",
    "accent2": "#00bfa5",
    "accent3": "#ff9100",
    "accent4": "#e040fb",
    "accent5": "#40c4ff",
    "drawdown": "#ff1744",
}

CATEGORY_COLORS = {
    "Trend": "#00d4ff",
    "Momentum": "#ff9100",
    "Oscillator": "#e040fb",
    "Volatility": "#00e676",
    "Support/Resistance": "#ffd740",
}


def plot_cumulative_returns(trades_df, save_dir=None):
    """
    Plot cumulative returns: Signal Strategy vs Buy-and-Hold with drawdown subplot.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1],
                                     sharex=True)

    dates = pd.to_datetime(trades_df["date"])

    # ─── Top: Cumulative Returns ───
    ax1.plot(dates, trades_df["cum_strategy"], color=COLORS["strategy"],
             linewidth=2, label="Signal Strategy", zorder=5)
    ax1.plot(dates, trades_df["cum_buyhold"], color=COLORS["buyhold"],
             linewidth=2, label="Buy & Hold", alpha=0.8, zorder=4)

    # Fill between
    ax1.fill_between(dates, trades_df["cum_strategy"], trades_df["cum_buyhold"],
                     where=trades_df["cum_strategy"] > trades_df["cum_buyhold"],
                     alpha=0.15, color=COLORS["strategy"], interpolate=True)
    ax1.fill_between(dates, trades_df["cum_strategy"], trades_df["cum_buyhold"],
                     where=trades_df["cum_strategy"] <= trades_df["cum_buyhold"],
                     alpha=0.15, color=COLORS["buyhold"], interpolate=True)

    # Crisis shading
    _add_crisis_shading(ax1)

    ax1.set_ylabel("Growth of $1", fontsize=12)
    ax1.set_title("Cumulative Returns: Signal Strategy vs Buy & Hold",
                  fontsize=16, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log") if trades_df["cum_buyhold"].max() > 5 else None

    # ─── Bottom: Drawdown ───
    cum_strat = trades_df["cum_strategy"]
    running_max = cum_strat.cummax()
    drawdown = (cum_strat - running_max) / running_max

    ax2.fill_between(dates, drawdown, 0, color=COLORS["drawdown"], alpha=0.5)
    ax2.plot(dates, drawdown, color=COLORS["drawdown"], linewidth=1)
    ax2.set_ylabel("Drawdown", fontsize=11)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    _add_crisis_shading(ax2)

    plt.tight_layout()
    _save_fig(fig, "cumulative_returns", save_dir)


def plot_weight_evolution(weight_history, save_dir=None, top_n=10):
    """
    Plot weight evolution over time for top N indicators.
    """
    # Build weight matrix
    years = sorted(weight_history.keys())
    all_indicators = weight_history[years[0]].index.tolist()

    weight_matrix = pd.DataFrame(index=years, columns=all_indicators, dtype=float)
    for year in years:
        for ind in all_indicators:
            if ind in weight_history[year].index:
                weight_matrix.loc[year, ind] = weight_history[year][ind]
            else:
                weight_matrix.loc[year, ind] = 0.0

    # Select top N by average weight
    avg_weights = weight_matrix.mean().nlargest(top_n)
    top_indicators = avg_weights.index.tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),
                                    gridspec_kw={"width_ratios": [2.5, 1]})

    # ─── Left: Line chart of weight evolution ───
    cmap = plt.cm.get_cmap("tab20", top_n)
    for i, ind in enumerate(top_indicators):
        ax1.plot(years, weight_matrix[ind].values, marker="o", markersize=4,
                 linewidth=2, label=ind, color=cmap(i), alpha=0.85)

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Weight", fontsize=12)
    ax1.set_title(f"Weight Evolution — Top {top_n} Indicators",
                  fontsize=16, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(years)
    ax1.tick_params(axis="x", rotation=45)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # ─── Right: Average weight bar chart ───
    colors = [cmap(i) for i in range(top_n)]
    bars = ax2.barh(range(top_n), avg_weights.values, color=colors, alpha=0.85)
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(top_indicators, fontsize=10)
    ax2.set_xlabel("Average Weight", fontsize=12)
    ax2.set_title("Average Weights (2006–2021)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, avg_weights.values):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=9, color="#e0e0e0")

    plt.tight_layout()
    _save_fig(fig, "weight_evolution", save_dir)


def plot_signal_distribution(trades_df, save_dir=None):
    """
    Plot monthly signal distribution showing BUY vs SELL signals over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])

    dates = pd.to_datetime(trades_df["date"])

    # ─── Top: Signal over time ───
    buy_mask = trades_df["signal"] == 1
    sell_mask = trades_df["signal"] == 0

    ax1.bar(dates[buy_mask], trades_df.loc[buy_mask, "weighted_sum"],
            width=25, color=COLORS["buy_signal"], alpha=0.7, label="BUY")
    ax1.bar(dates[sell_mask], trades_df.loc[sell_mask, "weighted_sum"],
            width=25, color=COLORS["sell_signal"], alpha=0.7, label="SELL")

    ax1.axhline(y=sc.ENSEMBLE_THRESHOLD, color=COLORS["neutral"],
                linestyle="--", alpha=0.7, label="Threshold (0.5)")
    if sc.ENABLE_DEAD_ZONE:
        ax1.axhspan(sc.DEAD_ZONE_LOW, sc.DEAD_ZONE_HIGH,
                    color=COLORS["neutral"], alpha=0.1, label="Dead Zone")

    ax1.set_ylabel("Weighted Signal Sum", fontsize=12)
    ax1.set_title("Monthly Signal Distribution",
                  fontsize=16, fontweight="bold", pad=15)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    _add_crisis_shading(ax1)

    # ─── Bottom: Yearly signal count ───
    yearly_signals = trades_df.groupby("year")["signal"].value_counts().unstack(fill_value=0)
    if 1 in yearly_signals.columns and 0 in yearly_signals.columns:
        years = yearly_signals.index
        x = np.arange(len(years))
        width = 0.35

        ax2.bar(x - width / 2, yearly_signals[1], width,
                color=COLORS["buy_signal"], alpha=0.8, label="BUY months")
        ax2.bar(x + width / 2, yearly_signals[0], width,
                color=COLORS["sell_signal"], alpha=0.8, label="SELL months")

        ax2.set_xticks(x)
        ax2.set_xticklabels(years, rotation=45)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_xlabel("Year", fontsize=12)
        ax2.legend(framealpha=0.9)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_fig(fig, "signal_distribution", save_dir)


def plot_category_contribution(weight_history, accuracy_history, save_dir=None):
    """
    Plot indicator category contribution to ensemble performance.
    """
    years = sorted(weight_history.keys())
    categories = sc.INDICATOR_CATEGORIES

    # Compute average weight per category per year
    cat_weights = {cat: [] for cat in categories}
    cat_accuracies = {cat: [] for cat in categories}

    for year in years:
        weights = weight_history[year]
        for cat, indicators in categories.items():
            active = [ind for ind in indicators if ind in weights.index]
            if active:
                cat_weights[cat].append(weights[active].sum())
            else:
                cat_weights[cat].append(0)

            if year in accuracy_history:
                accs = accuracy_history[year]
                active_acc = [ind for ind in indicators if ind in accs.index]
                if active_acc:
                    cat_accuracies[cat].append(accs[active_acc].mean())
                else:
                    cat_accuracies[cat].append(0.5)
            else:
                cat_accuracies[cat].append(0.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ─── Left: Stacked area chart of category weights ───
    cat_names = list(categories.keys())
    bottom = np.zeros(len(years))

    for cat in cat_names:
        vals = np.array(cat_weights[cat])
        color = CATEGORY_COLORS.get(cat, "#888888")
        ax1.fill_between(years, bottom, bottom + vals, alpha=0.7,
                         label=cat, color=color)
        ax1.plot(years, bottom + vals, color=color, linewidth=0.5, alpha=0.8)
        bottom += vals

    ax1.set_xlabel("Year", fontsize=12)
    ax1.set_ylabel("Total Weight", fontsize=12)
    ax1.set_title("Category Weight Distribution Over Time",
                  fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_xticks(years)
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # ─── Right: Average accuracy by category ───
    avg_accs = {cat: np.mean(cat_accuracies[cat]) for cat in cat_names}
    colors = [CATEGORY_COLORS.get(cat, "#888888") for cat in cat_names]

    bars = ax2.barh(cat_names, [avg_accs[c] for c in cat_names],
                    color=colors, alpha=0.85)
    ax2.axvline(x=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.7,
                label="Random (50%)")
    ax2.set_xlabel("Average Signal Accuracy", fontsize=12)
    ax2.set_title("Category Accuracy (2006–2021)",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")
    ax2.legend(framealpha=0.9)

    for bar, val in zip(bars, [avg_accs[c] for c in cat_names]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1%}", va="center", fontsize=10, color="#e0e0e0")

    plt.tight_layout()
    _save_fig(fig, "category_contribution", save_dir)


def plot_rolling_sharpe(trades_df, window=12, save_dir=None):
    """
    Plot 12-month rolling Sharpe ratio for strategy and buy-and-hold.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    dates = pd.to_datetime(trades_df["date"])

    for col, label, color in [
        ("portfolio_return", "Signal Strategy", COLORS["strategy"]),
        ("buy_hold_return", "Buy & Hold", COLORS["buyhold"]),
    ]:
        returns = trades_df[col]
        rolling_mean = returns.rolling(window=window, min_periods=6).mean()
        rolling_std = returns.rolling(window=window, min_periods=6).std()
        rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(12)

        ax.plot(dates, rolling_sharpe, color=color, linewidth=1.8,
                label=label, alpha=0.85)

    ax.axhline(y=0, color=COLORS["neutral"], linestyle="--", alpha=0.5)
    _add_crisis_shading(ax)

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"{window}-Month Rolling Sharpe", fontsize=12)
    ax.set_title(f"Rolling Sharpe Ratio ({window}-Month Window)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_fig(fig, "rolling_sharpe", save_dir)


def plot_accuracy_heatmap(accuracy_history, save_dir=None):
    """
    Plot Year × Indicator signal accuracy heatmap.
    """
    if not accuracy_history:
        print("  → No accuracy history to plot")
        return

    years = sorted(accuracy_history.keys())
    all_indicators = sorted(
        set().union(*(acc.index.tolist() for acc in accuracy_history.values()))
    )

    # Build matrix
    matrix = pd.DataFrame(index=years, columns=all_indicators, dtype=float)
    for year in years:
        for ind in all_indicators:
            if ind in accuracy_history[year].index:
                matrix.loc[year, ind] = accuracy_history[year][ind]
            else:
                matrix.loc[year, ind] = np.nan

    # Sort indicators by mean accuracy
    mean_acc = matrix.mean().sort_values(ascending=False)
    matrix = matrix[mean_acc.index]

    fig, ax = plt.subplots(figsize=(18, 10))

    # Custom colormap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix.values.astype(float), cmap=cmap, aspect="auto",
                   vmin=0.3, vmax=0.7)

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=9)

    # Add text annotations
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.iloc[i, j]
            if not np.isnan(val):
                text_color = "black" if 0.4 < val < 0.6 else "white"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=6, color=text_color)

    cbar = plt.colorbar(im, ax=ax, fraction=0.02)
    cbar.set_label("Signal Accuracy", fontsize=11)

    ax.set_title("Signal Accuracy Heatmap (Year × Indicator)",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Technical Indicator", fontsize=12)
    ax.set_ylabel("Year", fontsize=12)

    plt.tight_layout()
    _save_fig(fig, "accuracy_heatmap", save_dir)


def plot_yearly_comparison(yearly_metrics, save_dir=None):
    """
    Plot yearly return comparison: strategy vs buy-and-hold.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    years = yearly_metrics["year"].values
    x = np.arange(len(years))
    width = 0.35

    ax.bar(x - width / 2, yearly_metrics["strategy_return"] * 100, width,
           color=COLORS["strategy"], alpha=0.85, label="Signal Strategy")
    ax.bar(x + width / 2, yearly_metrics["buyhold_return"] * 100, width,
           color=COLORS["buyhold"], alpha=0.85, label="Buy & Hold")

    ax.axhline(y=0, color="white", linewidth=0.8, alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_ylabel("Annual Return (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_title("Annual Returns: Signal Strategy vs Buy & Hold",
                 fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")

    # Add accuracy labels above strategy bars
    for i, (_, row) in enumerate(yearly_metrics.iterrows()):
        if not np.isnan(row.get("accuracy", np.nan)):
            ax.text(i - width / 2, row["strategy_return"] * 100 + 1,
                    f"{row['accuracy']:.0%}", ha="center", fontsize=7,
                    color=COLORS["strategy"])

    plt.tight_layout()
    _save_fig(fig, "yearly_comparison", save_dir)


def generate_all_plots(results, save_dir=None):
    """
    Generate all visualization charts.

    Parameters
    ----------
    results : dict
        Output from ensemble.run_expanding_window_backtest().
    save_dir : str
        Directory to save plots. Default: signal_config.SIGNAL_RESULTS_DIR.
    """
    if save_dir is None:
        save_dir = sc.SIGNAL_RESULTS_DIR

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    trades_df = results["trades"]
    weight_history = results["weight_history"]
    accuracy_history = results["accuracy_history"]
    yearly_metrics = results["yearly_metrics"]

    # 1. Cumulative Returns
    print("  → Plotting cumulative returns...")
    plot_cumulative_returns(trades_df, save_dir)

    # 2. Weight Evolution
    print("  → Plotting weight evolution...")
    plot_weight_evolution(weight_history, save_dir)

    # 3. Signal Distribution
    print("  → Plotting signal distribution...")
    plot_signal_distribution(trades_df, save_dir)

    # 4. Category Contribution
    print("  → Plotting category contribution...")
    plot_category_contribution(weight_history, accuracy_history, save_dir)

    # 5. Rolling Sharpe
    print("  → Plotting rolling Sharpe ratio...")
    plot_rolling_sharpe(trades_df, save_dir=save_dir)

    # 6. Accuracy Heatmap
    print("  → Plotting accuracy heatmap...")
    plot_accuracy_heatmap(accuracy_history, save_dir)

    # 7. Yearly Comparison
    if len(yearly_metrics) > 0:
        print("  → Plotting yearly comparison...")
        plot_yearly_comparison(yearly_metrics, save_dir)

    print(f"\n  → All plots saved to {save_dir}")


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _add_crisis_shading(ax):
    """Add shaded regions for major market crises."""
    crises = [
        ("2007-10-01", "2009-03-01", "GFC"),
        ("2020-02-01", "2020-04-01", "COVID"),
        ("2015-08-01", "2016-02-01", "China"),
        ("2018-10-01", "2018-12-31", "Q4'18"),
    ]
    for start, end, label in crises:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.1, color="#ff5252")
        except Exception:
            pass


def _save_fig(fig, name, save_dir=None):
    """Save figure to disk."""
    if save_dir is None:
        save_dir = sc.SIGNAL_RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    → Saved: {path}")
