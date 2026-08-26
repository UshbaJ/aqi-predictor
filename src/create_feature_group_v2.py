"""
create_feature_group_v2.py

One-off migration script: builds the redesigned day-average targets
(target_24h/48h/72h) and pushes them to Hopsworks as
aqi_weather_features v2, leaving v1 untouched as a reference baseline.
"""

import os
import hopsworks
from dotenv import load_dotenv

from features import build_full_dataset

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()

df = build_full_dataset()
print(f"Built {len(df)} rows, {df.shape[1]} columns for v2 upload.")

fg = fs.get_or_create_feature_group(
    name="aqi_weather_features",
    version=2,
    primary_key=["datetime"],
    event_time="datetime",
    time_travel_format="HUDI",
    online_enabled=False,
    description=(
        "Hourly AQI + weather features for Bahawalpur, with lags/rolling "
        "stats and day-average targets (target_24h/48h/72h as non-overlapping "
        "24h window means, not point values)"
    ),
)

fg.insert(df, write_options={"wait_for_job": True})
print("v2 feature group created and populated.")


