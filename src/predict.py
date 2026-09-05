"""
predict.py

Inference script: loads the latest engineered features and the three
saved horizon models (Ridge, since it won at every horizon during CV)
to produce a 3-day-ahead AQI forecast, for a given city.

Usage:
    python src/predict.py --city bahawalpur
    python src/predict.py --city lahore
    python src/predict.py --city islamabad
"""

import argparse
import joblib

from features import load_features, FEATURE_COLS, HORIZONS, CITIES
from train import model_filename  # reuse the same naming convention as train.py

MODEL_TYPE = "ridge"  # ridge beat naive + rf at every horizon during CV


def load_models(city):
    models = {}
    for h in HORIZONS:
        path = model_filename(city, MODEL_TYPE, h)
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


def predict_next_3_days(city="bahawalpur"):
    """Returns a dict of {day_1: aqi, day_2: aqi, day_3: aqi} for dashboard use, for one city."""
    df, source = load_features(city=city, source="auto")
    latest_row = get_latest_feature_row(df)
    models = load_models(city)

    X_latest = latest_row[FEATURE_COLS].values.reshape(1, -1)
    results = {}
    for day_num, h in enumerate(HORIZONS, start=1):
        results[f"day_{day_num}"] = float(models[h].predict(X_latest)[0])
    return results, latest_row["datetime"], source


def main():
    parser = argparse.ArgumentParser(description="Forecast next 3 days of AQI for one city")
    parser.add_argument(
        "--city",
        choices=CITIES,
        default="bahawalpur",
        help="Which city to forecast for. Defaults to bahawalpur for backward compatibility.",
    )
    args = parser.parse_args()
    city = args.city

    print(f"[{city}] Loading features...")
    results, as_of, source = predict_next_3_days(city=city)
    print(f"[{city}] Using features from: {source}")
    print(f"[{city}] Forecasting from latest complete row: {as_of}\n")

    print(f"[{city}] 3-Day AQI Forecast (predicted daily average AQI):")
    print("=" * 50)
    for i, (day, aqi) in enumerate(results.items(), start=1):
        print(f"  {day.replace('_', ' ').title()}: {aqi:.1f} AQI")


if __name__ == "__main__":
    main()