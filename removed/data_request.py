import requests
import os
from dotenv import load_dotenv

def fetch_weather_data(api_key, city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return None

# API 호출 예시
load_dotenv("config.env")
API_KEY = os.getenv("API_KEY")
city = "Gwangju"
weather_data = fetch_weather_data(API_KEY, city)

