import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=api_key)
fs = project.get_feature_store()
fg = fs.get_feature_group("aqi_weather_features", version=2)

df = fg.read()
print(f"{len(df)} rows in v2")
print(df[["target_24h", "target_48h", "target_72h"]].describe())
print(f"Duplicate datetimes: {df['datetime'].duplicated().sum()}")