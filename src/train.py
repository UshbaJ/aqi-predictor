"""
train.py

Training pipeline for the AQI Predictor, using proper time-series
cross-validation (TimeSeriesSplit) instead of one fixed train/test split.
Supports multiple cities via --city.

Usage:
    python src/train.py --city bahawalpur
    python src/train.py --city lahore
    python src/train.py --city islamabad
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import FeatureLoadError, load_features, get_clean_dataset_for_horizon, FEATURE_COLS, HORIZONS, CITIES

MODEL_DIR = "src"
N_SPLITS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


def results_path_for_city(city):
    """Bahawalpur keeps its original unsuffixed filename for backward compatibility;
    other cities get suffixed filenames - same convention as features.py/feature_pipeline.py."""
    return "data/cv_validation_results.csv" if city == "bahawalpur" else f"data/cv_validation_results_{city}.csv"


def model_filename(city, model_type, horizon):
    """e.g. 'ridge_model_24h.pkl' for bahawalpur, 'ridge_model_lahore_24h.pkl' for others."""
    if city == "bahawalpur":
        return f"{MODEL_DIR}/{model_type}_model_{horizon}h.pkl"
    return f"{MODEL_DIR}/{model_type}_model_{city}_{horizon}h.pkl"


def naive_baseline_scores(y_true, current_aqi):
    """Naive persistence baseline: predict AQI stays the same."""
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, current_aqi)),
        "mae": mean_absolute_error(y_true, current_aqi),
        "r2": r2_score(y_true, current_aqi),
    }


def score_model(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def cross_validate_horizon(clean_df, target_col, horizon, n_splits=N_SPLITS):
    """
    Run TimeSeriesSplit CV for Ridge and RandomForest on one horizon.
    Returns a dict of fold-by-fold and mean/std results per model.
    """
    X = clean_df[FEATURE_COLS].values
    y = clean_df[target_col].values
    current_aqi = clean_df["aqi_epa"].values  # for naive baseline

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = {"naive": [], "ridge": [], "rf": []}

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        naive_test = current_aqi[test_idx]

        fold_results["naive"].append(naive_baseline_scores(y_test, naive_test))

        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        fold_results["ridge"].append(score_model(y_test, ridge.predict(X_test)))

        rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        fold_results["rf"].append(score_model(y_test, rf.predict(X_test)))

        print(f"  [{horizon}h] Fold {fold_idx}/{n_splits} "
              f"(train={len(train_idx)}, test={len(test_idx)}) done.")

    return fold_results


def summarize_fold_results(fold_results, horizon):
    """Print mean +/- std RMSE/MAE/R2 per model across folds."""
    print(f"\n{'='*70}")
    print(f"  {horizon}h HORIZON - {N_SPLITS}-FOLD TIME-SERIES CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"{'Model':<10}{'RMSE (mean+-std)':>22}{'MAE (mean+-std)':>22}{'R2 (mean+-std)':>18}")

    summary = {"horizon": horizon}
    for model_name in ["naive", "ridge", "rf"]:
        rmses = [f["rmse"] for f in fold_results[model_name]]
        maes = [f["mae"] for f in fold_results[model_name]]
        r2s = [f["r2"] for f in fold_results[model_name]]

        rmse_str = f"{np.mean(rmses):.2f} +- {np.std(rmses):.2f}"
        mae_str = f"{np.mean(maes):.2f} +- {np.std(maes):.2f}"
        r2_str = f"{np.mean(r2s):.3f} +- {np.std(r2s):.3f}"
        print(f"{model_name:<10}{rmse_str:>22}{mae_str:>22}{r2_str:>18}")

        summary[f"{model_name}_rmse_mean"] = np.mean(rmses)
        summary[f"{model_name}_rmse_std"] = np.std(rmses)
        summary[f"{model_name}_r2_mean"] = np.mean(r2s)
        summary[f"{model_name}_r2_std"] = np.std(r2s)

    best_model = min(["ridge", "rf"], key=lambda m: summary[f"{m}_rmse_mean"])
    naive_rmse = summary["naive_rmse_mean"]
    best_rmse = summary[f"{best_model}_rmse_mean"]

    if best_rmse < naive_rmse:
        improvement = (1 - best_rmse / naive_rmse) * 100
        print(f"\n  VERDICT: {best_model.upper()} beats naive by {improvement:.1f}% (mean RMSE across folds).")
    else:
        print(f"\n  VERDICT: neither model beats naive on average across folds.")

    return summary


def print_feature_importance(clean_df, target_col, horizon, top_n=10):
    """Fit a final RF on all data for this horizon, print top feature importances."""
    X = clean_df[FEATURE_COLS]
    y = clean_df[target_col]

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  Top {top_n} features for {horizon}h (Random Forest importance):")
    for feat, imp in importances.head(top_n).items():
        print(f"    {feat:<20} {imp:.3f}")

    return rf


def train_final_models(clean_df, target_col, horizon, city):
    """
    Train final Ridge + RF on ALL available data for this horizon
    (not a CV fold) and save to disk - these are the deployable models.
    """
    X = clean_df[FEATURE_COLS]
    y = clean_df[target_col]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y)
    joblib.dump(ridge, model_filename(city, "ridge", horizon))

    rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, model_filename(city, "rf", horizon))

    print(f"  [{city}] Saved final {horizon}h models to {MODEL_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Train AQI forecasting models for one city")
    parser.add_argument(
        "--city",
        choices=CITIES,
        default="bahawalpur",
        help="Which city to train models for. Defaults to bahawalpur for backward compatibility.",
    )
    args = parser.parse_args()
    city = args.city

    print(f"[{city}] Loading features (Hopsworks Feature Store, falling back to local CSVs)...")
    try:
        full_df, source = load_features(city=city, source="auto")
    except FeatureLoadError as e:
        print(f"WARNING: {e}")
        print(f"[{city}] Skipping retraining for this run — keeping existing model.")
        exit(0)  # exit cleanly, don't fail the workflow

    print(f"[{city}] Using features from: {source}\n")

    all_summaries = []

    for horizon in HORIZONS:
        print(f"\n[{city}] Processing {horizon}h horizon...")
        clean_df, target_col = get_clean_dataset_for_horizon(full_df, horizon)

        if len(clean_df) < N_SPLITS * 100:
            print(f"  Not enough clean rows ({len(clean_df)}) for {N_SPLITS}-fold CV - skipping.")
            continue

        fold_results = cross_validate_horizon(clean_df, target_col, horizon)
        summary = summarize_fold_results(fold_results, horizon)
        all_summaries.append(summary)

        print_feature_importance(clean_df, target_col, horizon)

        train_final_models(clean_df, target_col, horizon, city)

    if all_summaries:
        results_path = results_path_for_city(city)
        pd.DataFrame(all_summaries).to_csv(results_path, index=False)
        print(f"\n[{city}] Saved CV summary to {results_path}")


if __name__ == "__main__":
    main()