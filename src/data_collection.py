import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 29.3956, 71.6836  # Bahawalpur


def calculate_aqi(concentration, breakpoints):
    """
    Generic EPA AQI calculation for one pollutant.
    breakpoints: list of tuples (C_low, C_high, I_low, I_high)
    """
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= concentration <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (concentration - c_low) + i_low
            return round(aqi)
    return None  # concentration out of defined range


# EPA breakpoints (US standard) — (C_low, C_high, I_low, I_high)
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300), (250.5, 500.4, 301, 500)
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300), (425, 604, 301, 500)
]


def compute_overall_aqi(components):
    """Takes OpenWeather's components dict, returns EPA AQI (dominant pollutant)."""
    pm25_aqi = calculate_aqi(components["pm2_5"], PM25_BREAKPOINTS)
    pm10_aqi = calculate_aqi(components["pm10"], PM10_BREAKPOINTS)
    candidates = [v for v in [pm25_aqi, pm10_aqi] if v is not None]
    return max(candidates) if candidates else None


def fetch_current_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
    response = requests.get(url)
    data = response.json()

    record = data["list"][0]
    components = record["components"]
    timestamp = datetime.fromtimestamp(record["dt"])

    row = {
        "datetime": timestamp,
        "aqi_epa": compute_overall_aqi(components),
        "openweather_aqi": record["main"]["aqi"],
        **components
    }
    return row


def fetch_historical_data(days_back=30):
    end = int(datetime.now().timestamp())
    start = int((datetime.now() - timedelta(days=days_back)).timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start}&end={end}&appid={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        print("Error or no historical data:", data)
        return []

    rows = []
    for record in data["list"]:
        components = record["components"]
        timestamp = datetime.fromtimestamp(record["dt"])
        rows.append({
            "datetime": timestamp,
            "aqi_epa": compute_overall_aqi(components),
            "openweather_aqi": record["main"]["aqi"],
            **components
        })
    return rows


def save_to_csv(row, filepath="data/raw_aqi_data.csv"):
    df_new = pd.DataFrame([row])
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved row: {row}")


def save_many_to_csv(rows, filepath="data/raw_aqi_data.csv"):
    df_new = pd.DataFrame(rows)
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved {len(rows)} historical rows. Total in file: {len(df_combined)}")


if __name__ == "__main__":
    historical_rows = fetch_historical_data(days_back=30)
    if historical_rows:
        save_many_to_csv(historical_rows)

    current_row = fetch_current_data()
    save_to_csv(current_row)