"""
compute_shap.py

Computes SHAP feature importance for each horizon's deployed Ridge model,
using the already-saved model files from train.py. Saves mean |SHAP value|
per feature to CSV for the dashboard's "Why This Prediction" section.

Usage:
    python src/compute_shap.py
"""

import os
import joblib
import pandas as pd
import shap

from features import load_features, get_clean_dataset_for_horizon, FEATURE_COLS, HORIZONS

MODEL_DIR = "src"
OUTPUT_DIR = "data"
BACKGROUND_SAMPLE_SIZE = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_shap_for_horizon(df, horizon):
    clean_df, target_col = get_clean_dataset_for_horizon(df, horizon)
    X = clean_df[FEATURE_COLS]

    model_path = f"{MODEL_DIR}/ridge_model_{horizon}h.pkl"
    if not os.path.exists(model_path):
        print(f"  [{horizon}h] No saved model found at {model_path} - skipping.")
        return

    model = joblib.load(model_path)

    background = X.sample(min(BACKGROUND_SAMPLE_SIZE, len(X)), random_state=42)
    explainer = shap.LinearExplainer(model, background)
    shap_values = explainer(X)

    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "mean_abs_shap": abs(shap_values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    importance.to_csv(f"{OUTPUT_DIR}/shap_importance_{horizon}h.csv", index=False)
    print(f"  [{horizon}h] Top feature: {importance.iloc[0]['feature']} "
          f"({importance.iloc[0]['mean_abs_shap']:.2f})")


def main():
    print("Loading features...")
    df, source = load_features(source="auto")
    print(f"Using features from: {source}\n")

    for horizon in HORIZONS:
        compute_shap_for_horizon(df, horizon)

    print("\nSHAP computation complete. Results saved to data/shap_importance_{h}h.csv")


if __name__ == "__main__":
    main()