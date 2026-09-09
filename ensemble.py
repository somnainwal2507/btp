"""
Ensemble signal aggregation and row-based expanding-window backtesting.
Combines weighted indicator signals into a final BUY/SELL decision,
simulates trading performance with stop-loss/take-profit, and runs individual
strategy backtests for comparison.
"""

import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from signal_system import signal_config as sc
from signal_system.weight_optimizer import compute_signal_accuracy, compute_weights


def generate_ensemble_signal(signals_row, weights):
    """
    Compute weighted ensemble signal for a single time step.
    """
    common = signals_row.index.intersection(weights.index)
    if len(common) == 0:
        return -1, 0.5

    sig = signals_row[common].values.astype(float)
    w = weights[common].values.astype(float)

    w_sum = w.sum()
    if w_sum == 0:
        return -1, 0.5
    w = w / w_sum

    weighted_sum = np.nansum(sig * w)

    if getattr(sc, 'ENABLE_DEAD_ZONE', False):
        if weighted_sum > sc.DEAD_ZONE_HIGH:
            return 1, weighted_sum
        elif weighted_sum < sc.DEAD_ZONE_LOW:
            return 0, weighted_sum
        else:
            return -1, weighted_sum
    else:
        if weighted_sum > getattr(sc, 'ENSEMBLE_THRESHOLD', 0.5):
            return 1, weighted_sum
        else:
            return 0, weighted_sum


def run_row_based_backtest(signals_df, returns_series, close_prices, is_ensemble=True, strategy_name="Ensemble", indicator_weights=None):
    """
    Run a generic row-based expanding window backtest.
    """
    train_size = getattr(sc, 'TRAIN_PERIODS', 1000)
    test_size = getattr(sc, 'TEST_PERIODS', 100)
    
    total_rows = len(signals_df)
    
    trade_log = []
    weight_history = {}
    accuracy_history = {}
    
    prev_weights = None
    prev_signal = 1
    
    # SL/TP tracking
    entry_price = None
    
    indicator_names = signals_df.columns.tolist()
    
    for start_idx in range(train_size, total_rows, test_size):
        train_signals = signals_df.iloc[0:start_idx]
        train_returns = returns_series.iloc[0:start_idx]
        
        test_end = min(start_idx + test_size, total_rows)
        test_signals = signals_df.iloc[start_idx:test_end]
        test_returns = returns_series.iloc[start_idx:test_end]
        test_prices = close_prices.iloc[start_idx:test_end]
        
        iteration = start_idx // test_size
        
        if is_ensemble:
            shifted_returns = train_returns.shift(-1).dropna()
            eval_signals = train_signals.iloc[:len(shifted_returns)]
            
            accuracies = compute_signal_accuracy(eval_signals, shifted_returns)
            accuracy_history[iteration] = accuracies
            
            weights = compute_weights(
                accuracies,
                prev_weights=prev_weights,
                index_returns=train_returns,
                is_first_iteration=(iteration == train_size // test_size)
            )
            weight_history[iteration] = weights
            prev_weights = weights.copy()
        else:
            weights = pd.Series(1.0, index=[strategy_name]) if strategy_name in indicator_names else pd.Series(0.0)
            
        for i in range(len(test_signals)):
            idx = test_signals.index[i]
            signal_row = test_signals.iloc[i]
            actual_ret = test_returns.iloc[i] if not np.isnan(test_returns.iloc[i]) else 0.0
            current_price = test_prices.iloc[i]
            
            if is_ensemble:
                raw_signal, weighted_sum = generate_ensemble_signal(signal_row, weights)
            else:
                raw_signal = int(signal_row[strategy_name]) if strategy_name in signal_row and pd.notna(signal_row[strategy_name]) else -1
                weighted_sum = raw_signal
                
            # SL/TP logic
            forced_exit = False
            if getattr(sc, 'ENABLE_SL_TP', False) and prev_signal == 1 and entry_price is not None:
                price_change = (current_price - entry_price) / entry_price
                if price_change <= -sc.STOP_LOSS_PCT:
                    forced_exit = True
                    # exit at stop loss limit
                    actual_ret = max(actual_ret, -sc.STOP_LOSS_PCT)
                elif price_change >= sc.TAKE_PROFIT_PCT:
                    forced_exit = True
                    # exit at take profit limit
                    actual_ret = min(actual_ret, sc.TAKE_PROFIT_PCT)
            
            if forced_exit:
                final_signal = 0
            elif raw_signal == -1:
                final_signal = prev_signal
            else:
                final_signal = raw_signal
                
            if final_signal == 1 and prev_signal == 0:
                entry_price = current_price
            elif final_signal == 0:
                entry_price = None
                
            prev_signal = final_signal
            
            if final_signal == 1:
                portfolio_ret = actual_ret
            elif getattr(sc, 'SELL_BEHAVIOR', 'cash') == "short":
                portfolio_ret = -actual_ret
            elif getattr(sc, 'SELL_BEHAVIOR', 'cash') == "risk_free":
                portfolio_ret = 0.02 / 252 # daily approx
            else:
                portfolio_ret = 0.0
                
            predicted_up = (final_signal == 1)
            actual_up = (actual_ret > 0)
            is_correct = (predicted_up == actual_up)
            
            trade_log.append({
                "date": idx,
                "iteration": iteration,
                "signal": final_signal,
                "weighted_sum": weighted_sum,
                "actual_return": actual_ret,
                "portfolio_return": portfolio_ret,
                "buy_hold_return": actual_ret,
                "correct": is_correct if not np.isnan(actual_ret) else np.nan,
            })
            
    trades_df = pd.DataFrame(trade_log)
    if len(trades_df) > 0:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df = trades_df.sort_values("date").reset_index(drop=True)
        trades_df["cum_strategy"] = (1 + trades_df["portfolio_return"]).cumprod()
        trades_df["cum_buyhold"] = (1 + trades_df["buy_hold_return"]).cumprod()
        
    return {
        "trades": trades_df,
        "weight_history": weight_history,
        "accuracy_history": accuracy_history
    }


def run_all_strategies(signals_df, returns_series, close_prices):
    """
    Runs backtests for the ensemble and top individual strategies.
    Returns comparative results.
    """
    print("=" * 70)
    print("RUNNING ROW-BASED COMPARATIVE BACKTESTS")
    print("=" * 70)
    
    # 1. Ensemble
    print("  → Testing Ensemble Strategy...")
    ensemble_results = run_row_based_backtest(signals_df, returns_series, close_prices, is_ensemble=True)
    
    # 2. Individual strategies
    individual_results = {}
    core_indicators = [c for c in ['RSI', 'MACD', 'BB_PERCENT', 'SMAVG', 'EMAVG', 'ADX'] if c in signals_df.columns]
    
    for ind in core_indicators:
        print(f"  → Testing Individual Strategy: {ind}...")
        res = run_row_based_backtest(signals_df, returns_series, close_prices, is_ensemble=False, strategy_name=ind)
        individual_results[ind] = res['trades']
        
    return ensemble_results, individual_results

