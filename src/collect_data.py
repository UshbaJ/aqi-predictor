import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import argparse

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ============================================================
# CITIES
# ============================================================
# Add/edit cities here. Each entry needs a lat/lon pair.
CITIES = {
    "bahawalpur": {"lat": 29.3956, "lon": 71.6836},
    "lahore":     {"lat": 31.5497, "lon": 74.3436},
    "islamabad":  {"lat": 33.6844, "lon": 73.0479},
}

# OpenWeather's Air Pollution History API only goes back to 2020-11-27
EARLIEST_AQI_DATE = datetime(2020, 11, 27)


# ============================================================
# AQI / POLLUTION FUNCTIONS
# ============================================================

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


def fetch_current_data(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
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


def _fetch_aqi_window(lat, lon, start_dt, end_dt):
    """Single API call for one start/end window. Returns list of rows."""
    start = int(start_dt.timestamp())
    end = int(end_dt.timestamp())

    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={lat}&lon={lon}&start={start}&end={end}&appid={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "list" not in data:
        print(f"  WARNING: no data for {start_dt.date()} to {end_dt.date()}: {data}")
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


def fetch_historical_data(lat, lon, days_back=30, chunk_by_year=False):
    """
    Fetch historical AQI data for one city.
    If chunk_by_year=True, days_back is ignored and instead pulls in yearly
    chunks from EARLIEST_AQI_DATE (or today - days_back, whichever is later)
    up to now. This avoids timeouts/truncation on multi-year requests.
    """
    if not chunk_by_year:
        end = datetime.now()
        start = end - timedelta(days=days_back)
        return _fetch_aqi_window(lat, lon, start, end)

    # Yearly-chunked pull
    overall_start = max(EARLIEST_AQI_DATE, datetime.now() - timedelta(days=365 * 5))
    overall_end = datetime.now()

    all_rows = []
    chunk_start = overall_start
    chunk_num = 1
    while chunk_start < overall_end:
        chunk_end = min(chunk_start + timedelta(days=365), overall_end)
        print(f"  AQI chunk {chunk_num}: {chunk_start.date()} to {chunk_end.date()}...")

        rows = _fetch_aqi_window(lat, lon, chunk_start, chunk_end)
        print(f"    -> {len(rows)} rows")
        all_rows.extend(rows)

        chunk_start = chunk_end
        chunk_num += 1
        time.sleep(1)  # be polite to the API, avoid rate limits

    return all_rows


def save_to_csv(row, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_new = pd.DataFrame([row])
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.floor("h")
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"]).dt.floor("h")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", keep="last", inplace=True)
        df_combined.sort_values("datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved row: {row}")


def save_many_to_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        print("No AQI rows to save.")
        return
    df_new = pd.DataFrame(rows)
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.floor("h")
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"]).dt.floor("h")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", keep="last", inplace=True)
        df_combined.sort_values("datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved {len(rows)} historical rows. Total in file: {len(df_combined)}")


# ============================================================
# WEATHER FUNCTIONS
# ============================================================

def fetch_current_weather(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    row = {
        "datetime": datetime.fromtimestamp(data["dt"]),
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "wind_deg": data["wind"].get("deg", None),
    }
    return row


def _fetch_open_meteo_window(lat, lon, start_date, end_date):
    """Single API call for one start/end date window. Returns list of rows."""
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=ms"
        f"&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        print(f"  WARNING: no weather data for {start_date} to {end_date}: {data}")
        return []

    hourly = data["hourly"]
    rows = []
    for i in range(len(hourly["time"])):
        rows.append({
            "datetime": pd.to_datetime(hourly["time"][i]),
            "temp": hourly["temperature_2m"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "pressure": hourly["surface_pressure"][i],
            "wind_speed": hourly["wind_speed_10m"][i],
            "wind_deg": hourly["wind_direction_10m"][i],
        })
    return rows


def fetch_open_meteo_historical(lat, lon, days_back=30, chunk_by_year=False):
    """
    Free, no-API-key historical weather from Open-Meteo Archive API.
    If chunk_by_year=True, pulls in yearly chunks matching the AQI pull's
    date range for alignment (defaults to same 5-year / EARLIEST_AQI_DATE floor).
    """
    if not chunk_by_year:
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        return _fetch_open_meteo_window(lat, lon, start_date, end_date)

    overall_start = max(EARLIEST_AQI_DATE, datetime.now() - timedelta(days=365 * 5)).date()
    overall_end = datetime.now().date()

    all_rows = []
    chunk_start = overall_start
    chunk_num = 1
    while chunk_start < overall_end:
        chunk_end = min(chunk_start + timedelta(days=365), overall_end)
        print(f"  Weather chunk {chunk_num}: {chunk_start} to {chunk_end}...")

        rows = _fetch_open_meteo_window(lat, lon, chunk_start, chunk_end)
        print(f"    -> {len(rows)} rows")
        all_rows.extend(rows)

        chunk_start = chunk_end
        chunk_num += 1
        time.sleep(1)  # be polite to the API

    return all_rows


def save_weather_to_csv(rows, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not rows:
        print("No weather rows to save.")
        return
    df_new = pd.DataFrame(rows)
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.floor("h")
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"]).dt.floor("h")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", keep="last", inplace=True)
        df_combined.sort_values("datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved {len(rows)} weather rows. Total in file: {len(df_combined)}")


def save_current_weather_to_csv(row, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_new = pd.DataFrame([row])
    df_new["datetime"] = pd.to_datetime(df_new["datetime"]).dt.floor("h")
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df_existing["datetime"] = pd.to_datetime(df_existing["datetime"]).dt.floor("h")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.drop_duplicates(subset="datetime", keep="last", inplace=True)
        df_combined.sort_values("datetime", inplace=True)
    else:
        df_combined = df_new
    df_combined.to_csv(filepath, index=False)
    print(f"Saved weather row: {row}")


# ============================================================
# PER-CITY RUNNER
# ============================================================

def run_for_city(city_name, lat, lon, args):
    print(f"\n{'='*60}\nCITY: {city_name.upper()} (lat={lat}, lon={lon})\n{'='*60}")

    aqi_filepath = f"data/{city_name}/raw_aqi_data.csv"
    weather_filepath = f"data/{city_name}/raw_weather_data.csv"

    if args.backfill:
        print("=== Pulling historical AQI data (yearly chunks) ===")
        historical_rows = fetch_historical_data(lat, lon, chunk_by_year=True)
        save_many_to_csv(historical_rows, aqi_filepath)

        print("\n=== Pulling historical weather data (yearly chunks) ===")
        historical_weather = fetch_open_meteo_historical(lat, lon, chunk_by_year=True)
        save_weather_to_csv(historical_weather, weather_filepath)

    elif args.hours_back > 0:
        days_back = max(1, args.hours_back // 24 + 1)
        print(f"=== Pulling last {args.hours_back}h (~{days_back} day(s)) of AQI + weather ===")
        historical_rows = fetch_historical_data(lat, lon, days_back=days_back)
        save_many_to_csv(historical_rows, aqi_filepath)

        historical_weather = fetch_open_meteo_historical(lat, lon, days_back=days_back)
        save_weather_to_csv(historical_weather, weather_filepath)

    current_row = fetch_current_data(lat, lon)
    save_to_csv(current_row, aqi_filepath)

    current_weather = fetch_current_weather(lat, lon)
    print("Current weather:", current_weather)
    save_current_weather_to_csv(current_weather, weather_filepath)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AQI + weather data collection (multi-city)")
    parser.add_argument(
        "--city",
        choices=list(CITIES.keys()) + ["all"],
        default="all",
        help="Which city to pull data for. Default: all cities in CITIES dict.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Run the full 5-year yearly-chunked historical pull instead of "
             "just fetching recent data. Use this once for initial setup.",
    )
    parser.add_argument(
        "--hours-back",
        type=int,
        default=0,
        help="Pull the last N hours of history instead of just the current "
             "reading. Useful on CI runners with no local CSV history, so "
             "lag/rolling features can be computed correctly before "
             "pushing to the feature store. 0 = just the current reading.",
    )
    args = parser.parse_args()

    targets = CITIES.items() if args.city == "all" else [(args.city, CITIES[args.city])]

    for city_name, coords in targets:
        run_for_city(city_name, coords["lat"], coords["lon"], args)