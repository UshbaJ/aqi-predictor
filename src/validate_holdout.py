"""
validate_holdout.py

Genuine out-of-sample validation: trains Ridge on all data EXCLUDING the
last `HOLDOUT_DAYS`, then predicts on that held-out window. Saves
actual-vs-predicted results per horizon for the dashboard's validation chart.

Usage:
    python src/validate_holdout.py --city bahawalpur
    python src/validate_holdout.py --city lahore
    python src/validate_holdout.py --city islamabad
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from features import load_features, get_clean_dataset_for_horizon, FEATURE_COLS, HORIZONS, CITIES

HOLDOUT_DAYS = 90
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def holdout_output_path(city, horizon):
    """Bahawalpur keeps its original unsuffixed filename; other cities get suffixed -
    same convention as train.py/feature_pipeline.py."""
    if city == "bahawalpur":
        return f"{OUTPUT_DIR}/holdout_{horizon}h.csv"
    return f"{OUTPUT_DIR}/holdout_{city}_{horizon}h.csv"


def run_holdout_for_horizon(df, horizon, city):
    clean_df, target_col = get_clean_dataset_for_horizon(df, horizon)
    clean_df = clean_df.sort_values("datetime").reset_index(drop=True)

    cutoff = clean_df["datetime"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_df = clean_df[clean_df["datetime"] < cutoff]
    test_df = clean_df[clean_df["datetime"] >= cutoff]

    if len(test_df) < 10:
        print(f"  [{city}] [{horizon}h] Not enough holdout rows ({len(test_df)}) - skipping.")
        return None

    X_train, y_train = train_df[FEATURE_COLS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLS], test_df[target_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  [{city}] [{horizon}h] Holdout RMSE: {rmse:.2f} ({len(test_df)} rows, "
          f"train excludes last {HOLDOUT_DAYS} days)")

    result = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual": y_test.values,
        "predicted": y_pred,
    })
    result.to_csv(holdout_output_path(city, horizon), index=False)
    return rmse


def main():
    parser = argparse.ArgumentParser(description="Run holdout validation for one city")
    parser.add_argument(
        "--city",
        choices=CITIES,
        default="bahawalpur",
        help="Which city to validate. Defaults to bahawalpur for backward compatibility.",
    )
    args = parser.parse_args()
    city = args.city

    print(f"[{city}] Loading features...")
    df, source = load_features(city=city, source="auto")
    print(f"[{city}] Using features from: {source}\n")

    for horizon in HORIZONS:
        run_holdout_for_horizon(df, horizon, city)

    print(f"\n[{city}] Holdout validation complete. Results saved to {OUTPUT_DIR}/holdout_*.csv")


if __name__ == "__main__":
    main()