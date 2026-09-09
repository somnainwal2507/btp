"""
Main entry point for the Timeframe-Agnostic Technical Indicator Signal System.
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from signal_system import signal_config as sc
from signal_system.data_loader import load_and_prepare_data
from signal_system.indicator_signals import generate_all_signals
from signal_system.ensemble import run_all_strategies
from signal_system.performance import compute_performance_metrics

def run_signal_system():
    parser = argparse.ArgumentParser(description="Run Timeframe-Agnostic Trading System")
    parser.add_argument("--file", type=str, default="data/sp500_monthly.csv", help="Path to OHLCV Excel or CSV dataset")
    args = parser.parse_known_args()[0]

    start_time = time.time()
    print("╔" + "═" * 68 + "╗")
    print("║  TIMEFRAME-AGNOSTIC INDICATOR SIGNAL SYSTEM                       ║")
    print("╚" + "═" * 68 + "╝")

    if not os.path.exists(args.file):
        print(f"Dataset not found at {args.file}. Please provide a valid CSV/Excel file.")
        print("For testing purposes, falling back to cached sp500_index.parquet if available...")
        fallback = os.path.join(config.RAW_DIR, sc.INDEX_CACHE_FILE)
        if os.path.exists(fallback):
            df_fallback = pd.read_parquet(fallback)
            df_fallback.to_csv("data/sp500_monthly.csv", index=False)
            args.file = "data/sp500_monthly.csv"
        else:
            return

    # 1. Load Data
    try:
        df = load_and_prepare_data(args.file)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Generate Signals
    print("\n▶ STEP 2: GENERATING BINARY SIGNALS")
    indicator_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume", "ret"]]
    indicator_df = df[indicator_cols]
    close_prices = df["Close"]
    signals_df = generate_all_signals(indicator_df, close_prices)
    print(f"  → Generated signals for {len(signals_df.columns)} indicators")

    # 3. Prepare Returns
    returns = df['ret'].shift(-1)  # Next period's return
    returns.name = "forward_return"
    common_idx = signals_df.index.intersection(returns.dropna().index)
    signals_aligned = signals_df.loc[common_idx]
    returns_aligned = returns.loc[common_idx]
    close_prices_aligned = close_prices.loc[common_idx]
    
    # Use actual returns for backtesting (not forward returns)
    actual_returns = df['ret'].loc[common_idx]

    # 4. Backtesting
    ensemble_res, individual_res = run_all_strategies(signals_aligned, actual_returns, close_prices_aligned)

    # 5. Performance Evaluation & Export
    print("\n▶ STEP 5: PERFORMANCE EVALUATION")
    
    all_metrics = []
    
    if len(ensemble_res['trades']) > 0:
        ens_metrics = compute_performance_metrics(ensemble_res['trades'])
        ens_strat = ens_metrics["strategy"]
        ens_strat["Strategy"] = "Ensemble"
        all_metrics.append(ens_strat)
        
    for ind_name, trades_df in individual_res.items():
        if len(trades_df) > 0:
            m = compute_performance_metrics(trades_df)["strategy"]
            m["Strategy"] = ind_name
            all_metrics.append(m)

    if not all_metrics:
        print("No trades generated.")
        return

    metrics_df = pd.DataFrame(all_metrics)
    cols = ["Strategy", "win_rate", "sharpe_ratio", "max_drawdown", "total_return", "total_buy_signals", "total_sell_signals"]
    metrics_df = metrics_df[cols].set_index("Strategy")
    
    print("\nSummary Metrics:")
    print(metrics_df.to_string())

    # Export to Excel
    export_path = os.path.join(sc.SIGNAL_RESULTS_DIR, "strategy_comparison.xlsx")
    metrics_df.to_excel(export_path)
    print(f"\n  → Saved comparative metrics to {export_path}")

    # 6. Visualization
    print("\n▶ STEP 6: VISUALIZATION")
    
    # Plot Win Rate and Sharpe Ratio bar chart (Top 5 + Ensemble)
    # Filter valid Sharpe and sort by Sharpe
    plot_df = metrics_df.sort_values(by="sharpe_ratio", ascending=False)
    # Ensure Ensemble is included and get top 5
    if "Ensemble" in plot_df.index:
        top_strats = plot_df.drop("Ensemble").head(5)
        plot_df = pd.concat([plot_df.loc[["Ensemble"]], top_strats])
    else:
        plot_df = plot_df.head(6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Win Rate Plot
    bars = axes[0].bar(plot_df.index, plot_df["win_rate"] * 100, color=['gold' if x == "Ensemble" else 'steelblue' for x in plot_df.index])
    axes[0].set_title("Win Rate (%) Comparison")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].axhline(y=50, color='r', linestyle='--', alpha=0.6, label='50% Threshold')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Sharpe Plot
    axes[1].bar(plot_df.index, plot_df["sharpe_ratio"], color=['gold' if x == "Ensemble" else 'seagreen' for x in plot_df.index])
    axes[1].set_title("Sharpe Ratio Comparison")
    axes[1].set_ylabel("Sharpe Ratio")
    axes[1].axhline(y=0, color='k', linewidth=1)
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    chart_path = os.path.join(sc.SIGNAL_RESULTS_DIR, "win_rate_sharpe_comparison.png")
    plt.savefig(chart_path)
    print(f"  → Saved comparison chart to {chart_path}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    run_signal_system()
