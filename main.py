import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def step_download():
    """Step 1: Download all datasets."""
    from data.download import download_all
    return download_all()


def step_features():
    """Step 2: Build feature matrix (39 TIs + target)."""
    from features.builder import build_feature_matrix
    return build_feature_matrix()


def step_preprocess(df=None, ti_names=None):
    """Step 3: Preprocess features (exclusion, ffill, imputation)."""
    from preprocessing.pipeline import preprocess

    if df is None:
        # Load from disk
        path = os.path.join(config.RAW_DIR, "feature_matrix.parquet")
        df = pd.read_parquet(path)

        # Reconstruct TI names from columns
        id_cols = {"date", "ticker", "target", "excess_ret", "ret", "rf_rate",
                   "year", "month_end", "_month"}
        ti_set = set(config.TECHNICAL_INDICATORS)
        ti_names = [c for c in df.columns if c in ti_set]

    return preprocess(df, ti_names)


def step_train(df=None, feature_cols=None):
    """Step 4: Train all 5 models with expanding window."""
    from training.rolling_window import expanding_window_train

    if df is None:
        path = os.path.join(config.RAW_DIR, "preprocessed_data.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
        else:
            df, feature_cols = step_preprocess()

    if feature_cols is None:
        ti_set = set(config.TECHNICAL_INDICATORS)
        feature_cols = [c for c in df.columns if c in ti_set]

    return expanding_window_train(df, feature_cols)


def step_analyze(results, predictions, df, feature_cols):
    """Step 5: Run SHAP and uncertainty analyses."""
    print("\n" + "=" * 70)
    print("RUNNING ANALYSIS SUITE")
    print("=" * 70)

    # ─── SHAP Analysis ───
    try:
        from analysis.shap_analysis import run_shap_analysis
        from models.deep_nn import DeepNNModel

        # Need a trained Deep-NN model
        # Re-train on latest window for SHAP
        print("\n[ANALYSIS] Training Deep-NN for SHAP analysis...")
        df_sorted = df.copy()
        df_sorted["year"] = pd.to_datetime(df_sorted["date"]).dt.year

        train_mask = df_sorted["year"] <= (config.LAST_TEST_YEAR - 1)
        test_mask = df_sorted["year"] == config.LAST_TEST_YEAR

        X_train = df_sorted.loc[train_mask, feature_cols].values
        y_train = df_sorted.loc[train_mask, "target"].values
        X_test = df_sorted.loc[test_mask, feature_cols].values

        dnn = DeepNNModel()
        dnn.fit(X_train, y_train)

        shap_importance = run_shap_analysis(
            dnn, X_test, feature_cols, feature_cols
        )
    except Exception as e:
        print(f"[ANALYSIS] SHAP analysis failed: {e}")
        shap_importance = None

    # ─── Uncertainty Analysis ───
    try:
        from analysis.uncertainty import run_uncertainty_analysis
        uncertainty_results = run_uncertainty_analysis(predictions)
    except Exception as e:
        print(f"[ANALYSIS] Uncertainty analysis failed: {e}")
        uncertainty_results = None

    return {
        "shap": shap_importance,
        "uncertainty": uncertainty_results,
    }


def step_signals():
    """Step 6: Run the technical indicator signal system."""
    from signal_system.run_signal_system import run_signal_system
    return run_signal_system()


def run_all():
    """Run the complete pipeline end-to-end."""
    print("╔" + "═" * 68 + "╗")
    print("║  TECHNICAL INDICATORS ASSET PRICING FRAMEWORK                     ║")
    print("╚" + "═" * 68 + "╝")

    # Step 1: Download
    print("\n▶ STEP 1: DATA ACQUISITION")
    step_download()

    # Step 2: Feature engineering
    print("\n▶ STEP 2: FEATURE ENGINEERING")
    df, ti_names = step_features()

    # Step 3: Preprocessing
    print("\n▶ STEP 3: PREPROCESSING")
    df, final_ti = step_preprocess(df, ti_names)

    # Save preprocessed data
    preprocessed_path = os.path.join(config.RAW_DIR, "preprocessed_data.parquet")
    df.to_parquet(preprocessed_path, index=False)
    print(f"  → Saved preprocessed data to {preprocessed_path}")

    # Step 4: Train Models
    print("\n▶ STEP 4: TRAINING MODELS (Technical Indicators)")
    results, predictions = step_train(df, final_ti)

    # Step 5: Analysis
    print("\n▶ STEP 5: ANALYSIS & EXPLAINABILITY")
    analysis_results = step_analyze(
        results, predictions,
        df, final_ti,
    )

    # Final summary
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║  PIPELINE COMPLETE                                                ║")
    print("╚" + "═" * 68 + "╝")
    print(f"\nResults saved to: {config.RESULTS_DIR}")
    print(f"Files:")
    for f in os.listdir(config.RESULTS_DIR):
        fpath = os.path.join(config.RESULTS_DIR, f)
        size = os.path.getsize(fpath) / 1024
        print(f"  • {f} ({size:.1f} KB)")

    return results, predictions, analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Technical Indicators Asset Pricing Framework"
    )
    parser.add_argument("--all", action="store_true",
                        help="Run entire pipeline end-to-end")
    parser.add_argument("--download", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--features", action="store_true",
                        help="Build feature matrix")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing pipeline")
    parser.add_argument("--train", action="store_true",
                        help="Train all models")
    parser.add_argument("--analyze", action="store_true",
                        help="Run analysis suite")
    parser.add_argument("--signals", action="store_true",
                        help="Run technical indicator signal system")

    args = parser.parse_args()

    if args.all or not any(vars(args).values()):
        run_all()
    else:
        if args.download:
            step_download()
        if args.features:
            step_features()
        if args.preprocess:
            step_preprocess()
        if args.train:
            step_train()
        if args.analyze:
            # Need predictions for analysis
            print("Note: --analyze requires trained models. "
                  "Run --all for full pipeline.")
        if args.signals:
            step_signals()


if __name__ == "__main__":
    main()
