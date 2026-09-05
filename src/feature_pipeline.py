"""
feature_pipeline.py

Connects to Hopsworks Feature Store and pushes engineered features
(from features.py) into a versioned Feature Group, per city.

Usage:
    python src/feature_pipeline.py --city bahawalpur
    python src/feature_pipeline.py --city lahore
    python src/feature_pipeline.py --city islamabad
"""

import os
import argparse
import hopsworks
from dotenv import load_dotenv

from features import build_full_dataset, CITIES

load_dotenv()

FEATURE_GROUP_VERSION = 2
PRIMARY_KEY = ["datetime"]
EVENT_TIME = "datetime"


def feature_group_name(city):
    """Same naming convention as features.py's load_from_feature_store():
    bahawalpur keeps the original unsuffixed name, other cities get suffixed."""
    return "aqi_weather_features" if city == "bahawalpur" else f"aqi_weather_features_{city}"


def connect():
    """Log in to Hopsworks using the API key from .env, return the project's feature store."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in .env - add it before running this script.")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print(f"Connected to Hopsworks project: {project.name}")
    return fs


def prepare_dataframe(city):
    """
    Build the full engineered dataset for one city from features.py.
    Hopsworks needs clean column names and consistent dtypes, and
    doesn't accept NaN in primary key / event time columns, so we
    keep NaNs in feature columns (they get dropped per-horizon later
    during training) but ensure datetime is clean.
    """
    df = build_full_dataset(city=city)
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Hopsworks feature/column names must be lowercase, no special chars.
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    return df


HOPSWORKS_TO_PANDAS_DTYPE = {
    "bigint": "int64",
    "int": "int32",
    "double": "float64",
    "float": "float32",
    "boolean": "bool",
}


def align_dtypes_to_schema(df, fg):
    """
    Cast each column in df to match the feature group's already-established
    schema (fg.schema), so incremental inserts of any size don't fail with
    a type mismatch depending on whether NaN happened to be present.
    """
    df = df.copy()
    schema_types = {f.name: f.type for f in fg.schema}

    for col in df.columns:
        hw_type = schema_types.get(col)
        target_dtype = HOPSWORKS_TO_PANDAS_DTYPE.get(hw_type)
        if target_dtype is None:
            continue
        if target_dtype.startswith("int") and df[col].isna().any():
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(target_dtype)

    return df


def create_or_get_feature_group(fs, city):
    """Create the feature group for this city if it doesn't exist yet, or get the existing one."""
    fg_name = feature_group_name(city)
    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=FEATURE_GROUP_VERSION,
        description=f"Hourly AQI + weather features for {city.title()}, "
                     f"with lags/rolling stats and 24h/48h/72h day-average targets",
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        online_enabled=False,
        time_travel_format="HUDI",
    )
    return fg


def run_for_city(city):
    fs = connect()
    df = prepare_dataframe(city)
    print(f"[{city}] Prepared {len(df)} rows, {len(df.columns)} columns for upload.")

    fg = create_or_get_feature_group(fs, city)
    fg_name = feature_group_name(city)
    print(f"[{city}] Feature group '{fg_name}' (v{FEATURE_GROUP_VERSION}) ready.")
    if fg.schema:  # only true for a feature group that already has data/columns established
        existing_cols = {f.name for f in fg.schema}
        new_cols = [c for c in df.columns if c not in existing_cols]
        if new_cols:
            from hsfs.feature import Feature
            new_features = [Feature(name=c, type="double") for c in new_cols]
            print(f"[{city}] Registering new columns in feature group schema: {new_cols}")
            fg.append_features(new_features)
            fg = fs.get_feature_group(name=fg_name, version=FEATURE_GROUP_VERSION)
    else:
        print(f"[{city}] New feature group with no existing schema — "
              f"skipping column registration; first insert will establish the schema.")

    df = align_dtypes_to_schema(df, fg)
    print(f"[{city}] Aligned dataframe dtypes to feature group schema. Inserting data...")

    fg.insert(df)
    print(f"[{city}] Insert complete. Check the Hopsworks UI (Feature Store) to confirm the data landed.")


def main():
    parser = argparse.ArgumentParser(description="Push engineered features to Hopsworks Feature Store")
    parser.add_argument(
        "--city",
        choices=CITIES,
        required=True,
        help="Which city's feature group to create/update.",
    )
    args = parser.parse_args()
    run_for_city(args.city)


if __name__ == "__main__":
    main()