import pandas as pd


def load_raw_data(filepath="data/raw_aqi_data.csv"):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset="datetime")
    return df


def add_time_features(df):
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    return df


def add_lag_features(df):
    df["aqi_lag_1h"] = df["aqi_epa"].shift(1)
    df["aqi_lag_24h"] = df["aqi_epa"].shift(24)
    df["pm2_5_lag_1h"] = df["pm2_5"].shift(1)
    df["pm2_5_lag_24h"] = df["pm2_5"].shift(24)
    df["pm10_lag_1h"] = df["pm10"].shift(1)
    return df


def add_rolling_features(df):
    df["aqi_rolling_mean_6h"] = df["aqi_epa"].rolling(window=6).mean()
    return df


def add_target(df, horizon_hours=1):
    df[f"target_aqi_next_{horizon_hours}h"] = df["aqi_epa"].shift(-horizon_hours)
    return df


def build_feature_dataset(filepath="data/raw_aqi_data.csv", horizon_hours=1):
    df = load_raw_data(filepath)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df, horizon_hours)

    # Drop rows with NaNs created by lag/rolling/target shifting
    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    features_df = build_feature_dataset()
    print(features_df.shape)
    print(features_df.head())
    features_df.to_csv("data/features_data.csv", index=False)
    print("Saved to data/features_data.csv")