"""
Performance metrics for the signal-based trading system.
Computes Sharpe ratio, max drawdown, win rate, and other statistics.
"""

import numpy as np
import pandas as pd

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_system import signal_config as sc


def compute_performance_metrics(trades_df):
    """
    Compute comprehensive performance metrics for the signal strategy
    and the buy-and-hold benchmark.

    Parameters
    ----------
    trades_df : DataFrame
        Output from ensemble.run_expanding_window_backtest().
        Must contain: portfolio_return, buy_hold_return, correct, signal, date.

    Returns
    -------
    metrics : dict
        Dictionary of performance metrics for both strategy and benchmark.
    """
    strat_returns = trades_df["portfolio_return"].values
    bh_returns = trades_df["buy_hold_return"].values

    metrics = {}

    # ─── Strategy Metrics ───
    metrics["strategy"] = _compute_return_metrics(strat_returns, "Strategy")
    metrics["strategy"]["directional_accuracy"] = trades_df["correct"].mean()
    metrics["strategy"]["total_buy_signals"] = (trades_df["signal"] == 1).sum()
    metrics["strategy"]["total_sell_signals"] = (trades_df["signal"] == 0).sum()
    metrics["strategy"]["pct_time_invested"] = (
        (trades_df["signal"] == 1).sum() / len(trades_df)
    )

    # ─── Buy-and-Hold Metrics ───
    metrics["buyhold"] = _compute_return_metrics(bh_returns, "Buy & Hold")

    # ─── Relative Metrics ───
    metrics["relative"] = {
        "excess_total_return": (
            metrics["strategy"]["total_return"] -
            metrics["buyhold"]["total_return"]
        ),
        "excess_sharpe": (
            metrics["strategy"]["sharpe_ratio"] -
            metrics["buyhold"]["sharpe_ratio"]
        ),
        "drawdown_improvement": (
            metrics["strategy"]["max_drawdown"] -
            metrics["buyhold"]["max_drawdown"]
        ),
    }

    return metrics


def _compute_return_metrics(returns, name="Strategy"):
    """
    Compute standard return-based performance metrics.

    Parameters
    ----------
    returns : array-like
        Monthly returns.
    name : str
        Label for the strategy.

    Returns
    -------
    dict of metrics.
    """
    returns = np.array(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return {k: 0.0 for k in [
            "name", "total_return", "annualized_return", "annualized_vol",
            "sharpe_ratio", "max_drawdown", "win_rate", "profit_factor",
            "avg_monthly_return", "best_month", "worst_month",
            "positive_months", "negative_months", "total_months",
        ]}

    # Cumulative return
    cumulative = np.cumprod(1 + returns)
    total_return = cumulative[-1] - 1

    # Annualized return (geometric)
    n_years = len(returns) / 12
    if n_years > 0 and cumulative[-1] > 0:
        annualized_return = cumulative[-1] ** (1 / n_years) - 1
    else:
        annualized_return = 0.0

    # Annualized volatility
    annualized_vol = np.std(returns, ddof=1) * np.sqrt(12)

    # Sharpe ratio (assuming 0 risk-free for simplicity, monthly)
    if np.std(returns, ddof=1) > 0:
        sharpe = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(12)
    else:
        sharpe = 0.0

    # Maximum drawdown
    cumulative_series = pd.Series(cumulative)
    running_max = cumulative_series.cummax()
    drawdown = (cumulative_series - running_max) / running_max
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    return {
        "name": name,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_monthly_return": np.mean(returns),
        "best_month": np.max(returns),
        "worst_month": np.min(returns),
        "positive_months": int((returns > 0).sum()),
        "negative_months": int((returns < 0).sum()),
        "total_months": len(returns),
    }


def format_metrics_table(metrics):
    """
    Format performance metrics as a pretty comparison table.

    Parameters
    ----------
    metrics : dict
        Output from compute_performance_metrics().

    Returns
    -------
    table_str : str
        Formatted table string.
    df : DataFrame
        Comparison DataFrame.
    """
    strat = metrics["strategy"]
    bh = metrics["buyhold"]

    rows = [
        ("Total Return", f"{strat['total_return']:+.2%}", f"{bh['total_return']:+.2%}"),
        ("Annualized Return", f"{strat['annualized_return']:+.2%}", f"{bh['annualized_return']:+.2%}"),
        ("Annualized Volatility", f"{strat['annualized_vol']:.2%}", f"{bh['annualized_vol']:.2%}"),
        ("Sharpe Ratio", f"{strat['sharpe_ratio']:.3f}", f"{bh['sharpe_ratio']:.3f}"),
        ("Max Drawdown", f"{strat['max_drawdown']:.2%}", f"{bh['max_drawdown']:.2%}"),
        ("Win Rate", f"{strat['win_rate']:.1%}", f"{bh['win_rate']:.1%}"),
        ("Profit Factor", f"{strat['profit_factor']:.2f}", f"{bh['profit_factor']:.2f}"),
        ("Best Month", f"{strat['best_month']:+.2%}", f"{bh['best_month']:+.2%}"),
        ("Worst Month", f"{strat['worst_month']:+.2%}", f"{bh['worst_month']:+.2%}"),
        ("Avg Monthly Return", f"{strat['avg_monthly_return']:+.4f}", f"{bh['avg_monthly_return']:+.4f}"),
        ("Directional Accuracy", f"{strat.get('directional_accuracy', 0):.1%}", "N/A"),
        ("% Time Invested", f"{strat.get('pct_time_invested', 1):.1%}", "100.0%"),
        ("Total Months", f"{strat['total_months']}", f"{bh['total_months']}"),
    ]

    df = pd.DataFrame(rows, columns=["Metric", "Signal Strategy", "Buy & Hold"])

    # Build formatted string
    header = f"{'Metric':<25s} {'Signal Strategy':>18s} {'Buy & Hold':>18s}"
    separator = "─" * 63
    lines = [separator, header, separator]
    for _, row in df.iterrows():
        lines.append(f"{row['Metric']:<25s} {row['Signal Strategy']:>18s} {row['Buy & Hold']:>18s}")
    lines.append(separator)

    return "\n".join(lines), df


def compute_rolling_metrics(trades_df, window=12):
    """
    Compute rolling Sharpe ratio and drawdown.

    Parameters
    ----------
    trades_df : DataFrame
        Trade log.
    window : int
        Rolling window in months.

    Returns
    -------
    rolling_df : DataFrame with rolling_sharpe, rolling_drawdown columns.
    """
    returns = trades_df["portfolio_return"].copy()

    rolling_mean = returns.rolling(window=window, min_periods=6).mean()
    rolling_std = returns.rolling(window=window, min_periods=6).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(12)

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.rolling(window=window, min_periods=1).max()
    rolling_dd = (cumulative - rolling_max) / rolling_max

    rolling_df = pd.DataFrame({
        "date": trades_df["date"],
        "rolling_sharpe": rolling_sharpe.values,
        "rolling_drawdown": rolling_dd.values,
        "cumulative_return": cumulative.values,
    })

    return rolling_df
