import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_features(filepath):
    return pd.read_csv(filepath, parse_dates=["datetime"])


def time_based_split(df, test_fraction=0.2):
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def get_features_and_target(df, target_col):
    exclude_cols = ["datetime", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return df[feature_cols], df[target_col], feature_cols


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  {label}: RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def run_for_horizon(horizon):
    target_col = f"target_aqi_next_{horizon}h"
    df = load_features(f"data/features_data_{horizon}h.csv")
    train_df, test_df = time_based_split(df)

    print(f"\n=== Horizon: {horizon}h (train={len(train_df)}, test={len(test_df)}) ===")

    X_train, y_train, feature_cols = get_features_and_target(train_df, target_col)
    X_test, y_test, _ = get_features_and_target(test_df, target_col)

    # Naive baseline
    naive_metrics = evaluate(test_df[target_col], test_df["aqi_epa"], label="Naive")

    # Ridge — sweep alpha, keep the best
    best_ridge = None
    best_ridge_metrics = None
    for alpha in [1.0, 10.0, 50.0, 100.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate(y_test, y_pred, label=f"Ridge (alpha={alpha})")
        if best_ridge_metrics is None or metrics["rmse"] < best_ridge_metrics["rmse"]:
            best_ridge = model
            best_ridge_metrics = metrics
            best_ridge_metrics["alpha"] = alpha

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    rf_metrics = evaluate(y_test, y_pred_rf, label="Random Forest")

    joblib.dump(best_ridge, f"src/ridge_model_{horizon}h.pkl")
    joblib.dump(rf_model, f"src/rf_model_{horizon}h.pkl")

    return naive_metrics, best_ridge_metrics, rf_metrics, rf_model, feature_cols


if __name__ == "__main__":
    results = {}
    feature_importances = {}

    for horizon in [1, 24, 72]:
        naive_m, ridge_m, rf_m, rf_model, feature_cols = run_for_horizon(horizon)
        results[horizon] = {"naive": naive_m, "ridge": ridge_m, "rf": rf_m}
        feature_importances[horizon] = dict(zip(feature_cols, rf_model.feature_importances_))

    print("\n=== Summary (RMSE, lower is better) ===")
    print(f"{'Horizon':<10}{'Naive':<10}{'Ridge':<20}{'RandomForest':<15}")
    for horizon, r in results.items():
        ridge_label = f"{r['ridge']['rmse']:.2f} (a={r['ridge']['alpha']})"
        print(f"{horizon}h{'':<7}{r['naive']['rmse']:<10.2f}{ridge_label:<20}{r['rf']['rmse']:<15.2f}")

    print("\n=== Random Forest Top 5 Features per Horizon ===")
    for horizon, importances in feature_importances.items():
        top5 = sorted(importances.items(), key=lambda x: -x[1])[:5]
        print(f"\n{horizon}h:")
        for feat, imp in top5:
            print(f"  {feat}: {imp:.3f}")