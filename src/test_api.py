import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
lat, lon = 29.3956, 71.6836  # Bahawalpur

url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

response = requests.get(url)
print("Status code:", response.status_code)
print(response.json())