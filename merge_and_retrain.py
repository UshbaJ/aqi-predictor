"""
merge_and_retrain.py

Purpose: Validate whether adding weather features actually improves
24h/72h AQI forecasts before investing time in a 5-year historical pull.

Steps:
1. Load raw_aqi_data.csv and raw_weather_data.csv
2. Merge on datetime (inner join - only keep timestamps present in both)
3. Engineer basic time-series features (lags, rolling stats, time-of-day)
4. Build 24h and 72h ahead prediction targets
5. Train/test split respecting time order (no shuffling - time series!)
6. Train Ridge and RandomForest for each horizon
7. Compare RMSE/MAE/R2 against a naive baseline (persistence: predict AQI stays the same)
8. Save results to a summary CSV + print a clear report
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

AQI_PATH = "data/raw_aqi_data.csv"
WEATHER_PATH = "data/raw_weather_data.csv"
OUTPUT_DIR = "src"
RESULTS_PATH = "data/validation_results.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD + MERGE
# ============================================================

def load_and_merge():
    aqi = pd.read_csv(AQI_PATH, parse_dates=["datetime"])
    weather = pd.read_csv(WEATHER_PATH, parse_dates=["datetime"])

    aqi = aqi.sort_values("datetime").drop_duplicates(subset="datetime")
    weather = weather.sort_values("datetime").drop_duplicates(subset="datetime")

    merged = pd.merge(aqi, weather, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    print(f"AQI rows: {len(aqi)} | Weather rows: {len(weather)} | Merged (matched) rows: {len(merged)}")
    if len(merged) < 100:
        print("WARNING: very few matched rows - check that both CSVs cover overlapping dates.")

    return merged


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    # Lag features (past AQI values - critical for time-series forecasting)
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi_epa"].shift(lag)

    # Rolling stats on AQI
    df["aqi_roll_mean_6h"] = df["aqi_epa"].rolling(6).mean()
    df["aqi_roll_std_6h"] = df["aqi_epa"].rolling(6).std()

    # Weather features are already columns: temp, humidity, pressure, wind_speed, wind_deg
    # Lag weather slightly too, since current weather is known but helps stabilize
    for lag in [1, 3]:
        df[f"temp_lag_{lag}h"] = df["temp"].shift(lag)
        df[f"humidity_lag_{lag}h"] = df["humidity"].shift(lag)
        df[f"wind_speed_lag_{lag}h"] = df["wind_speed"].shift(lag)

    return df


def build_targets(df, horizons=(24, 72)):
    df = df.copy()
    for h in horizons:
        df[f"target_{h}h"] = df["aqi_epa"].shift(-h)
    return df


# ============================================================
# 3. TRAIN / EVALUATE PER HORIZON
# ============================================================

FEATURE_COLS = [
    "aqi_epa", "hour", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_6h", "aqi_roll_std_6h",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "temp_lag_1h", "temp_lag_3h", "humidity_lag_1h", "humidity_lag_3h",
    "wind_speed_lag_1h", "wind_speed_lag_3h",
]


def evaluate_horizon(df, horizon, test_frac=0.2):
    target_col = f"target_{horizon}h"
    cols_needed = FEATURE_COLS + [target_col]
    clean = df.dropna(subset=cols_needed).reset_index(drop=True)

    if len(clean) < 50:
        print(f"  [{horizon}h] Not enough clean rows ({len(clean)}) after dropping NaNs - skipping.")
        return None

    split_idx = int(len(clean) * (1 - test_frac))
    train = clean.iloc[:split_idx]
    test = clean.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train[target_col]
    X_test, y_test = test[FEATURE_COLS], test[target_col]

    results = {"horizon": horizon, "train_rows": len(train), "test_rows": len(test)}

    # --- Naive baseline: predict current AQI persists ---
    naive_pred = test["aqi_epa"].values
    results["naive_rmse"] = np.sqrt(mean_squared_error(y_test, naive_pred))
    results["naive_mae"] = mean_absolute_error(y_test, naive_pred)
    results["naive_r2"] = r2_score(y_test, naive_pred)

    # --- Ridge ---
    ridge = Ridge(alpha=10.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)
    results["ridge_rmse"] = np.sqrt(mean_squared_error(y_test, ridge_pred))
    results["ridge_mae"] = mean_absolute_error(y_test, ridge_pred)
    results["ridge_r2"] = r2_score(y_test, ridge_pred)
    joblib.dump(ridge, f"{OUTPUT_DIR}/ridge_model_{horizon}h_weather.pkl")

    # --- Random Forest ---
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results["rf_rmse"] = np.sqrt(mean_squared_error(y_test, rf_pred))
    results["rf_mae"] = mean_absolute_error(y_test, rf_pred)
    results["rf_r2"] = r2_score(y_test, rf_pred)
    joblib.dump(rf, f"{OUTPUT_DIR}/rf_model_{horizon}h_weather.pkl")

    return results


# ============================================================
# 4. REPORT
# ============================================================

def print_report(results):
    print(f"\n{'='*60}")
    print(f"  {results['horizon']}h HORIZON  (train={results['train_rows']} rows, test={results['test_rows']} rows)")
    print(f"{'='*60}")
    print(f"{'Model':<12}{'RMSE':>10}{'MAE':>10}{'R2':>10}")
    print(f"{'Naive':<12}{results['naive_rmse']:>10.2f}{results['naive_mae']:>10.2f}{results['naive_r2']:>10.3f}")
    print(f"{'Ridge':<12}{results['ridge_rmse']:>10.2f}{results['ridge_mae']:>10.2f}{results['ridge_r2']:>10.3f}")
    print(f"{'RF':<12}{results['rf_rmse']:>10.2f}{results['rf_mae']:>10.2f}{results['rf_r2']:>10.3f}")

    best_rmse = min(results["ridge_rmse"], results["rf_rmse"])
    if best_rmse < results["naive_rmse"]:
        improvement = (1 - best_rmse / results["naive_rmse"]) * 100
        print(f"\n  RESULT: Weather features HELPED. Best model beats naive by {improvement:.1f}% (RMSE).")
    else:
        print(f"\n  RESULT: Weather features did NOT beat naive baseline yet. "
              f"Likely still a data-volume issue (only ~30 days) rather than a feature issue.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    merged = load_and_merge()
    featured = engineer_features(merged)
    targeted = build_targets(featured, horizons=(24, 72))

    all_results = []
    for horizon in (24, 72):
        print(f"\nEvaluating {horizon}h horizon...")
        res = evaluate_horizon(targeted, horizon)
        if res:
            print_report(res)
            all_results.append(res)

    if all_results:
        pd.DataFrame(all_results).to_csv(RESULTS_PATH, index=False)
        print(f"\nSaved summary to {RESULTS_PATH}")
    else:
        print("\nNo horizon had enough data to evaluate. Check your CSVs.")
