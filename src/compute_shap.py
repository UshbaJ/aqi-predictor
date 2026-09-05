"""
compute_shap.py

Computes SHAP feature importance for each horizon's deployed Ridge model,
using the already-saved model files from train.py. Saves mean |SHAP value|
per feature to CSV for the dashboard's "Why This Prediction" section.

Usage:
    python src/compute_shap.py --city bahawalpur
    python src/compute_shap.py --city lahore
    python src/compute_shap.py --city islamabad
"""

import os
import argparse
import joblib
import pandas as pd
import shap

from features import load_features, get_clean_dataset_for_horizon, FEATURE_COLS, HORIZONS, CITIES
from train import model_filename  # reuse the same naming convention as train.py/predict.py

OUTPUT_DIR = "data"
BACKGROUND_SAMPLE_SIZE = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


def shap_output_path(city, horizon):
    """Bahawalpur keeps its original unsuffixed filename; other cities get suffixed."""
    if city == "bahawalpur":
        return f"{OUTPUT_DIR}/shap_importance_{horizon}h.csv"
    return f"{OUTPUT_DIR}/shap_importance_{city}_{horizon}h.csv"


def compute_shap_for_horizon(df, horizon, city):
    clean_df, target_col = get_clean_dataset_for_horizon(df, horizon)
    X = clean_df[FEATURE_COLS]

    model_path = model_filename(city, "ridge", horizon)
    if not os.path.exists(model_path):
        print(f"  [{city}] [{horizon}h] No saved model found at {model_path} - skipping.")
        return

    model = joblib.load(model_path)

    background = X.sample(min(BACKGROUND_SAMPLE_SIZE, len(X)), random_state=42)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(X)

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(shap_output_path(city, horizon), index=False)
    print(f"  [{city}] [{horizon}h] Top feature: {importance.iloc[0]['feature']} "
          f"({importance.iloc[0]['mean_abs_shap']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP feature importance for one city")
    parser.add_argument(
        "--city",
        choices=CITIES,
        default="bahawalpur",
        help="Which city to compute SHAP importance for. Defaults to bahawalpur for backward compatibility.",
    )
    args = parser.parse_args()
    city = args.city

    print(f"[{city}] Loading features...")
    df, source = load_features(city=city, source="auto")
    print(f"[{city}] Using features from: {source}\n")

    for horizon in HORIZONS:
        compute_shap_for_horizon(df, horizon, city)

    print(f"\n[{city}] SHAP computation complete. Results saved to {OUTPUT_DIR}/shap_importance_*.csv")


if __name__ == "__main__":
    main()