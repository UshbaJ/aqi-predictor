"""
features.py

Reusable data loading and feature engineering for the AQI Predictor.
Used by train.py (and later by the Hopsworks feature pipeline).

Keeping this separate from training logic means:
- The same feature definitions are used consistently every time
- Feature engineering can be tested/inspected independently
- When we move to Hopsworks, this becomes the feature pipeline
"""

import pandas as pd

AQI_PATH = "data/raw_aqi_data.csv"
WEATHER_PATH = "data/raw_weather_data.csv"

# Features used by all models. Keep this list as the single source of truth -
# train.py and any future inference code should import FEATURE_COLS from here
# rather than redefining it, so features never drift out of sync.
FEATURE_COLS = [
    "aqi_epa", "hour", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_6h", "aqi_roll_std_6h",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "temp_lag_1h", "temp_lag_3h", "humidity_lag_1h", "humidity_lag_3h",
    "wind_speed_lag_1h", "wind_speed_lag_3h",
]

HORIZONS = (24, 72)


def load_and_merge(aqi_path=AQI_PATH, weather_path=WEATHER_PATH):
    """Load raw AQI and weather CSVs, merge on datetime (inner join)."""
    aqi = pd.read_csv(aqi_path, parse_dates=["datetime"])
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])

    aqi = aqi.sort_values("datetime").drop_duplicates(subset="datetime")
    weather = weather.sort_values("datetime").drop_duplicates(subset="datetime")

    merged = pd.merge(aqi, weather, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    print(f"AQI rows: {len(aqi)} | Weather rows: {len(weather)} | Merged rows: {len(merged)}")
    if len(merged) < 100:
        print("WARNING: very few matched rows - check that both CSVs cover overlapping dates.")

    return merged


def engineer_features(df):
    """Add lag, rolling, and time-based features. Does not drop NaNs (caller decides)."""
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek

    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi_epa"].shift(lag)

    df["aqi_roll_mean_6h"] = df["aqi_epa"].rolling(6).mean()
    df["aqi_roll_std_6h"] = df["aqi_epa"].rolling(6).std()

    for lag in [1, 3]:
        df[f"temp_lag_{lag}h"] = df["temp"].shift(lag)
        df[f"humidity_lag_{lag}h"] = df["humidity"].shift(lag)
        df[f"wind_speed_lag_{lag}h"] = df["wind_speed"].shift(lag)

    return df


def build_targets(df, horizons=HORIZONS):
    """Add target_{h}h columns - AQI value h hours ahead."""
    df = df.copy()
    for h in horizons:
        df[f"target_{h}h"] = df["aqi_epa"].shift(-h)
    return df


def get_clean_dataset_for_horizon(df, horizon):
    """Drop rows with NaN in any feature or the target for this horizon."""
    target_col = f"target_{horizon}h"
    cols_needed = FEATURE_COLS + [target_col]
    return df.dropna(subset=cols_needed).reset_index(drop=True), target_col


def build_full_dataset():
    """Convenience entrypoint: load, merge, engineer features, build all targets."""
    merged = load_and_merge()
    featured = engineer_features(merged)
    targeted = build_targets(featured, horizons=HORIZONS)
    return targeted
