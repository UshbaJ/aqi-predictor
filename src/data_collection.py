import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT, LON = 29.3956, 71.6836  # Bahawalpur


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


def save_many_to_csv(rows, filepath="data/raw_aqi_data.csv"):
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

def fetch_current_weather():
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
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


def fetch_historical_weather(days_back=30):
    """
    Uses OpenWeather's timemachine endpoint (One Call 3.0) - one day at a time.
    NOTE: This endpoint may require billing setup even for free-tier usage.
    Not currently used in __main__ — kept as a fallback/reference.
    Prefer fetch_open_meteo_historical() below (free, no billing risk).
    """
    rows = []
    for day_offset in range(days_back, 0, -1):
        timestamp = int((datetime.now() - timedelta(days=day_offset)).timestamp())
        url = (
            f"http://api.openweathermap.org/data/3.0/onecall/timemachine"
            f"?lat={LAT}&lon={LON}&dt={timestamp}&appid={API_KEY}&units=metric"
        )
        response = requests.get(url)
        data = response.json()

        if "data" not in data and "current" not in data:
            print(f"Skipping day_offset={day_offset}: {data}")
            continue

        record = data.get("current", data.get("data", [{}])[0])
        rows.append({
            "datetime": datetime.fromtimestamp(record["dt"]),
            "temp": record.get("temp"),
            "humidity": record.get("humidity"),
            "pressure": record.get("pressure"),
            "wind_speed": record.get("wind_speed"),
            "wind_deg": record.get("wind_deg"),
        })
    return rows


def fetch_open_meteo_historical(days_back=30):
    """
    Free, no-API-key historical weather from Open-Meteo Archive API.
    Primary source for historical weather backfill — no billing risk,
    unlike OpenWeather's One Call 3.0 timemachine endpoint.
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m"
        f"&wind_speed_unit=ms"
        f"&timezone=auto"
    )
    response = requests.get(url)
    data = response.json()

    if "hourly" not in data:
        print("Error fetching Open-Meteo data:", data)
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


def save_weather_to_csv(rows, filepath="data/raw_weather_data.csv"):
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


def save_current_weather_to_csv(row, filepath="data/raw_weather_data.csv"):
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
# MAIN
# ============================================================

if __name__ == "__main__":
    # --- Pollution data (existing, working) ---
    historical_rows = fetch_historical_data(days_back=30)
    if historical_rows:
        save_many_to_csv(historical_rows)

    current_row = fetch_current_data()
    save_to_csv(current_row)

    # --- Weather data: Open-Meteo historical backfill (free, no billing risk) ---
    historical_weather = fetch_open_meteo_historical(days_back=30)
    if historical_weather:
        save_weather_to_csv(historical_weather)

    # --- Current weather (OpenWeather, real-time) ---
    current_weather = fetch_current_weather()
    print("Current weather:", current_weather)
    save_current_weather_to_csv(current_weather)