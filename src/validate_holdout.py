"""
validate_holdout.py

Genuine out-of-sample validation: trains Ridge on all data EXCLUDING the
last `HOLDOUT_DAYS`, then predicts on that held-out window. Saves
actual-vs-predicted results per horizon for the dashboard's validation chart.

Usage:
    python src/validate_holdout.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from features import load_features, get_clean_dataset_for_horizon, FEATURE_COLS, HORIZONS

HOLDOUT_DAYS = 90
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_holdout_for_horizon(df, horizon):
    clean_df, target_col = get_clean_dataset_for_horizon(df, horizon)
    clean_df = clean_df.sort_values("datetime").reset_index(drop=True)

    cutoff = clean_df["datetime"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train_df = clean_df[clean_df["datetime"] < cutoff]
    test_df = clean_df[clean_df["datetime"] >= cutoff]

    if len(test_df) < 10:
        print(f"  [{horizon}h] Not enough holdout rows ({len(test_df)}) - skipping.")
        return None

    X_train, y_train = train_df[FEATURE_COLS], train_df[target_col]
    X_test, y_test = test_df[FEATURE_COLS], test_df[target_col]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  [{horizon}h] Holdout RMSE: {rmse:.2f} ({len(test_df)} rows, "
          f"train excludes last {HOLDOUT_DAYS} days)")

    result = pd.DataFrame({
        "datetime": test_df["datetime"].values,
        "actual": y_test.values,
        "predicted": y_pred,
    })
    result.to_csv(f"{OUTPUT_DIR}/holdout_{horizon}h.csv", index=False)
    return rmse


def main():
    print("Loading features...")
    df, source = load_features(source="auto")
    print(f"Using features from: {source}\n")

    for horizon in HORIZONS:
        run_holdout_for_horizon(df, horizon)

    print("\nHoldout validation complete. Results saved to data/holdout_{h}h.csv")


if __name__ == "__main__":
    main()