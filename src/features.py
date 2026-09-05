"""
features.py

Reusable data loading and feature engineering for the AQI Predictor.
Used by train.py (and later by the Hopsworks feature pipeline).
Supports multiple cities via the `city` parameter.
"""

import pandas as pd

CITIES = ["bahawalpur", "lahore", "islamabad"]

FEATURE_COLS = [
    "aqi_epa", "hour", "day_of_week",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_6h", "aqi_roll_std_6h",
    "temp", "humidity", "pressure", "wind_speed", "wind_deg",
    "temp_lag_1h", "temp_lag_3h", "humidity_lag_1h", "humidity_lag_3h",
    "wind_speed_lag_1h", "wind_speed_lag_3h",
]

HORIZONS = (24, 48, 72)


def _paths_for_city(city):
    return f"data/{city}/raw_aqi_data.csv", f"data/{city}/raw_weather_data.csv"


def load_and_merge(city="bahawalpur", aqi_path=None, weather_path=None):
    """Load raw AQI and weather CSVs for a city, merge on datetime (inner join)."""
    default_aqi_path, default_weather_path = _paths_for_city(city)
    aqi_path = aqi_path or default_aqi_path
    weather_path = weather_path or default_weather_path

    aqi = pd.read_csv(aqi_path, parse_dates=["datetime"])
    weather = pd.read_csv(weather_path, parse_dates=["datetime"])

    aqi = aqi.sort_values("datetime").drop_duplicates(subset="datetime")
    weather = weather.sort_values("datetime").drop_duplicates(subset="datetime")

    merged = pd.merge(aqi, weather, on="datetime", how="inner")
    merged = merged.sort_values("datetime").reset_index(drop=True)

    print(f"[{city}] AQI rows: {len(aqi)} | Weather rows: {len(weather)} | Merged rows: {len(merged)}")
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
    """
    Add target_{h}h columns - mean AQI over the day-window ending at h hours ahead.
    Windows are non-overlapping 24h blocks: target_24h = mean(aqi[t+1:t+24]),
    target_48h = mean(aqi[t+25:t+48]), target_72h = mean(aqi[t+49:t+72]).
    """
    df = df.copy()
    for h in horizons:
        window_mean = df["aqi_epa"].rolling(window=24).mean()
        df[f"target_{h}h"] = window_mean.shift(-h)
    return df


def get_clean_dataset_for_horizon(df, horizon):
    """Drop rows with NaN in any feature or the target for this horizon."""
    target_col = f"target_{horizon}h"
    cols_needed = FEATURE_COLS + [target_col]
    return df.dropna(subset=cols_needed).reset_index(drop=True), target_col


def build_full_dataset(city="bahawalpur"):
    """Convenience entrypoint: load, merge, engineer features, build all targets for one city."""
    merged = load_and_merge(city=city)
    featured = engineer_features(merged)
    targeted = build_targets(featured, horizons=HORIZONS)
    return targeted


def load_from_feature_store(city="bahawalpur", max_retries=3, retry_delay=10):
    """
    Attempt to read the engineered feature set back from the Hopsworks
    Feature Store for a given city (aqi_weather_features_{city}, v2).
    Retries a few times before giving up, since Hopsworks' Arrow Flight
    service occasionally drops the connection mid-read.
    """
    import os
    import time
    import hopsworks
    from dotenv import load_dotenv

    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("HOPSWORKS_API_KEY")
    except Exception:
        pass

    if not api_key:
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")

    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in Streamlit secrets or .env file.")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()

    fg_name = "aqi_weather_features" if city == "bahawalpur" else f"aqi_weather_features_{city}"
    fg = fs.get_feature_group(name=fg_name, version=2)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            df = fg.read(read_options={"use_hive": True})
            df = df.sort_values("datetime").reset_index(drop=True)
            return df
        except Exception as e:
            last_error = e
            print(f"[{city}] Hopsworks read attempt {attempt}/{max_retries} failed "
                  f"({type(e).__name__}); retrying in {retry_delay}s...")
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise last_error


class FeatureLoadError(Exception):
    """Raised when neither Hopsworks nor local CSVs can supply training data."""
    pass


def load_features(city="bahawalpur", source="auto"):
    """
    Main entrypoint for train.py. Tries the Hopsworks Feature Store first
    (source="auto" or "hopsworks"), falls back to local CSVs on any failure
    (source="auto" or "local" explicitly forces local).

    On CI runners, local CSVs won't exist (data/ is gitignored), so a total
    failure raises FeatureLoadError with a clear message instead of an
    unhandled traceback - callers (train.py) should catch this and skip
    that day's retrain rather than crashing the whole job.
    """
    if source == "local":
        return build_full_dataset(city=city), "local"

    if source in ("auto", "hopsworks"):
        try:
            print(f"[{city}] Attempting to load features from Hopsworks Feature Store...")
            df = load_from_feature_store(city=city)
            print(f"[{city}] Loaded {len(df)} rows from Hopsworks Feature Store.")
            return df, "hopsworks"
        except Exception as hopsworks_error:
            if source == "hopsworks":
                raise
            print(f"[{city}] Hopsworks Feature Store read failed "
                  f"({type(hopsworks_error).__name__}); falling back to local CSVs.")
            try:
                return build_full_dataset(city=city), "local"
            except FileNotFoundError as csv_error:
                raise FeatureLoadError(
                    f"[{city}] No data source available: Hopsworks failed "
                    f"({type(hopsworks_error).__name__}: {hopsworks_error}) and "
                    f"local CSVs are missing ({csv_error}). Skipping retrain."
                ) from csv_error

    raise ValueError(f"Unknown source: {source!r}. Use 'auto', 'hopsworks', or 'local'.")

