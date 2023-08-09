
import requests

bike_data = {
    'season': 1,
    'mnth': 1,
    'holiday': 0,
    'weekday': 6, 
    'workingday': 0, 
    'weathersit': 2,
    'temp': 0.344167, 
    'atemp': 0.363625,
    'hum': 0.805833,
    'windspeed': 0.160446
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=bike_data)
print(response.json())

