"""
feature_pipeline.py

Connects to Hopsworks Feature Store and pushes engineered features
(from features.py) into a versioned Feature Group.

This is the bridge between local development (CSV -> pandas) and the
"real" feature store setup your project spec requires. After this runs
successfully once, train.py can be updated to read from Hopsworks
instead of recomputing from local CSVs every time.

Usage:
    python src/feature_pipeline.py
"""

import os
import hopsworks
from dotenv import load_dotenv

from features import build_full_dataset

load_dotenv()

FEATURE_GROUP_NAME = "aqi_weather_features"
FEATURE_GROUP_VERSION = 1
PRIMARY_KEY = ["datetime"]
EVENT_TIME = "datetime"


def connect():
    """Log in to Hopsworks using the API key from .env, return the project's feature store."""
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise ValueError("HOPSWORKS_API_KEY not found in .env - add it before running this script.")

    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    print(f"Connected to Hopsworks project: {project.name}")
    return fs


def prepare_dataframe():
    """
    Build the full engineered dataset from features.py.
    Hopsworks needs clean column names and consistent dtypes, and
    doesn't accept NaN in primary key / event time columns, so we
    keep NaNs in feature columns (they get dropped per-horizon later
    during training) but ensure datetime is clean.
    """
    df = build_full_dataset()
    df = df.dropna(subset=["datetime"]).reset_index(drop=True)

    # Hopsworks feature/column names must be lowercase, no special chars.
    # Our columns are already lowercase snake_case, so no renaming needed -
    # but double check nothing slipped through with capitals or dots.
    df.columns = [c.lower().replace(".", "_") for c in df.columns]

    return df


# Maps Hopsworks/Hudi column types (as reported by fg.schema) to the pandas
# dtype we should cast to before inserting, so a batch's dtype always
# matches the schema locked in on first upload - regardless of whether
# this particular batch happens to contain NaN in a given column.
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
    Columns with NaN can't be cast to a non-nullable int type - those are
    left as float (NaN-safe); if Hopsworks still rejects them, that
    reflects a genuine data issue worth investigating rather than
    something to silently paper over.
    """
    df = df.copy()
    schema_types = {f.name: f.type for f in fg.schema}

    for col in df.columns:
        hw_type = schema_types.get(col)
        target_dtype = HOPSWORKS_TO_PANDAS_DTYPE.get(hw_type)
        if target_dtype is None:
            continue  # string/timestamp/unknown - leave as-is
        if target_dtype.startswith("int") and df[col].isna().any():
            df[col] = df[col].astype("float64")
        else:
            df[col] = df[col].astype(target_dtype)

    return df


def create_or_get_feature_group(fs):
    """Create the feature group if it doesn't exist yet, or get the existing one."""
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Bahawalpur, with lags/rolling stats and 24h/72h targets",
        primary_key=PRIMARY_KEY,
        event_time=EVENT_TIME,
        online_enabled=False,  # offline-only is fine for this project; no real-time serving needed
        time_travel_format="HUDI",  # explicit - avoids ambiguous DELTA auto-detection failing on missing deltalake lib
    )
    return fg


def main():
    fs = connect()
    df = prepare_dataframe()
    print(f"Prepared {len(df)} rows, {len(df.columns)} columns for upload.")

    fg = create_or_get_feature_group(fs)
    print(f"Feature group '{FEATURE_GROUP_NAME}' (v{FEATURE_GROUP_VERSION}) ready.")

    df = align_dtypes_to_schema(df, fg)
    print("Aligned dataframe dtypes to feature group schema. Inserting data...")

    fg.insert(df)
    print("Insert complete. Check the Hopsworks UI (Feature Store) to confirm the data landed.")


if __name__ == "__main__":
    main()
