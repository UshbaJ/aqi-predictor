"""
predict.py

Inference script: loads the latest engineered features and the three
saved horizon models (Ridge, since it won at every horizon during CV)
to produce a 3-day-ahead AQI forecast.

Usage:
    python src/predict.py
"""

import joblib

from features import load_features, FEATURE_COLS, HORIZONS

MODEL_DIR = "src"
MODEL_TYPE = "ridge"  # ridge beat naive + rf at every horizon during CV


def load_models():
    models = {}
    for h in HORIZONS:
        path = f"{MODEL_DIR}/{MODEL_TYPE}_model_{h}h.pkl"
        models[h] = joblib.load(path)
    return models


def get_latest_feature_row(df):
    """
    Return the most recent row with no NaN in any feature column -
    this is what we feed into each horizon's model to forecast forward.
    """
    clean = df.dropna(subset=FEATURE_COLS)
    if clean.empty:
        raise ValueError("No rows with complete feature data found - check the feature pipeline.")
    return clean.sort_values("datetime").iloc[-1]


def predict_next_3_days():
    """Returns a dict of {day_1: aqi, day_2: aqi, day_3: aqi} for dashboard use."""
    df, source = load_features(source="auto")
    latest_row = get_latest_feature_row(df)
    models = load_models()

    X_latest = latest_row[FEATURE_COLS].values.reshape(1, -1)
    results = {}
    for day_num, h in enumerate(HORIZONS, start=1):
        results[f"day_{day_num}"] = float(models[h].predict(X_latest)[0])
    return results, latest_row["datetime"], source


def main():
    print("Loading features...")
    results, as_of, source = predict_next_3_days()
    print(f"Using features from: {source}")
    print(f"Forecasting from latest complete row: {as_of}\n")

    print("3-Day AQI Forecast (predicted daily average AQI):")
    print("=" * 50)
    for i, (day, aqi) in enumerate(results.items(), start=1):
        print(f"  {day.replace('_', ' ').title()}: {aqi:.1f} AQI")


if __name__ == "__main__":
    main()