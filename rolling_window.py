"""
Rolling / expanding window training pipeline.
Train 1990-2005, Test 2006 → Train 1990-2006, Test 2007 → ... → Train 1990-2020, Test 2021
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

from models.linear import OLSModel, LASSOModel
from models.tree import DecisionTreeModel, RandomForestModel
from models.deep_nn import DeepNNModel


def compute_r2_oos(y_true, y_pred, y_hist_mean=None):
    """
    Out-of-sample R² = 1 - Σ(r - r̂)² / Σ(r - r̄)²
    where r̄ is the historical mean from the training set.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)

    if y_hist_mean is not None:
        ss_tot = np.sum((y_true - y_hist_mean) ** 2)
    else:
        ss_tot = np.sum(y_true ** 2)

    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def get_model_instances():
    """Create fresh instances of all 5 models."""
    return {
        "OLS": OLSModel(),
        "LASSO": LASSOModel(),
        "DecisionTree": DecisionTreeModel(),
        "RandomForest": RandomForestModel(),
        "DeepNN": DeepNNModel(),
    }


def expanding_window_train(df, feature_cols, target_col="target"):
    """
    Expanding window training and evaluation.

    Iteration 1: Train 1990–2005, Test 2006
    Iteration 2: Train 1990–2006, Test 2007
    ...
    Final: Train 1990–2020, Test 2021

    Parameters
    ----------
    df : DataFrame with features and target, plus [date, ticker] columns.
    feature_cols : list of feature column names.
    target_col : name of the target column.

    Returns
    -------
    results : dict of {model_name: DataFrame with per-year metrics}
    predictions : dict of {model_name: DataFrame with predictions per test year}
    """
    print("=" * 70)
    print("EXPANDING WINDOW TRAINING")
    print("=" * 70)

    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    test_years = list(range(config.FIRST_TEST_YEAR, config.LAST_TEST_YEAR + 1))
    print(f"  → Test years: {test_years[0]} to {test_years[-1]} "
          f"({len(test_years)} iterations)")

    results = {name: [] for name in get_model_instances().keys()}
    predictions = {name: [] for name in get_model_instances().keys()}

    for test_year in test_years:
        train_end_year = test_year - 1
        print(f"\n{'─' * 50}")
        print(f"Iteration: Train {config.TRAIN_START_YEAR}–{train_end_year}, "
              f"Test {test_year}")
        print(f"{'─' * 50}")

        # Split data
        train_mask = df["year"] <= train_end_year
        test_mask = df["year"] == test_year

        train_data = df[train_mask]
        test_data = df[test_mask]

        if len(test_data) == 0:
            print(f"  → No test data for {test_year}, skipping.")
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[target_col].values
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values

        # Historical mean for R²_OOS benchmark
        y_hist_mean = np.mean(y_train)

        print(f"  → Train: {len(train_data)} samples | Test: {len(test_data)} samples")

        # Train and evaluate each model
        models = get_model_instances()

        for model_name, model in models.items():
            start_time = time.time()
            try:
                # Fit
                model.fit(X_train, y_train)

                # Predict
                y_pred = model.predict(X_test)

                # Metrics
                r2_oos = compute_r2_oos(y_test, y_pred, y_hist_mean)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                elapsed = time.time() - start_time

                results[model_name].append({
                    "test_year": test_year,
                    "R2_OOS": r2_oos,
                    "MSE": mse,
                    "MAE": mae,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "time_sec": elapsed,
                })

                # Store predictions
                pred_df = test_data[["date", "ticker"]].copy()
                pred_df["y_true"] = y_test
                pred_df["y_pred"] = y_pred
                pred_df["model"] = model_name
                pred_df["test_year"] = test_year
                predictions[model_name].append(pred_df)

                print(f"  {model_name:15s} → R²_OOS: {r2_oos:+.4f} | "
                      f"MSE: {mse:.6f} | MAE: {mae:.6f} | "
                      f"Time: {elapsed:.1f}s")

            except Exception as e:
                print(f"  {model_name:15s} → ERROR: {e}")
                results[model_name].append({
                    "test_year": test_year,
                    "R2_OOS": np.nan,
                    "MSE": np.nan,
                    "MAE": np.nan,
                    "train_size": len(train_data),
                    "test_size": len(test_data),
                    "time_sec": 0,
                })

    # Convert to DataFrames
    results_dfs = {}
    predictions_dfs = {}

    for model_name in results:
        results_dfs[model_name] = pd.DataFrame(results[model_name])
        if predictions[model_name]:
            predictions_dfs[model_name] = pd.concat(
                predictions[model_name], ignore_index=True
            )

    # Save results
    save_results(results_dfs, predictions_dfs)

    return results_dfs, predictions_dfs


def save_results(results_dfs, predictions_dfs):
    """Save results and predictions to disk."""
    print("\n[TRAINING] Saving results...")

    # Aggregate results table
    all_results = []
    for model_name, df in results_dfs.items():
        df_copy = df.copy()
        df_copy["model"] = model_name
        all_results.append(df_copy)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        path = os.path.join(config.RESULTS_DIR, "model_results.csv")
        combined.to_csv(path, index=False)
        print(f"  → Saved results to {path}")

        # Print summary table
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        summary = combined.groupby("model").agg({
            "R2_OOS": ["mean", "std"],
            "MSE": "mean",
            "MAE": "mean",
            "time_sec": "sum",
        }).round(6)
        print(summary)

    # Save predictions
    for model_name, df in predictions_dfs.items():
        path = os.path.join(config.RESULTS_DIR, f"predictions_{model_name}.csv")
        df.to_csv(path, index=False)

    print("  → All predictions saved.")
