import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_features(filepath="data/features_data.csv"):
    df = pd.read_csv(filepath, parse_dates=["datetime"])
    return df


def time_based_split(df, test_fraction=0.2):
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def get_features_and_target(df, target_col="target_aqi_next_1h"):
    exclude_cols = ["datetime", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def evaluate(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {label} ---")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE:  {mae:.2f}")
    print(f"R2:   {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def naive_baseline(test_df, target_col="target_aqi_next_1h"):
    # Naive: predict next hour's AQI = current hour's AQI
    y_true = test_df[target_col]
    y_pred = test_df["aqi_epa"]
    return evaluate(y_true, y_pred, label="Naive Baseline (predict = current AQI)")


def train_ridge(train_df, test_df, target_col="target_aqi_next_1h"):
    X_train, y_train, feature_cols = get_features_and_target(train_df, target_col)
    X_test, y_test, _ = get_features_and_target(test_df, target_col)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred, label="Ridge Regression")

    return model, metrics, feature_cols


if __name__ == "__main__":
    df = load_features()
    train_df, test_df = time_based_split(df)

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    naive_baseline(test_df)
    model, metrics, feature_cols = train_ridge(train_df, test_df)

    joblib.dump(model, "src/ridge_model.pkl")
    print("\nModel saved to src/ridge_model.pkl")