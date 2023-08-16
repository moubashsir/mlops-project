import json
import requests
from deepdiff import DeepDiff

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
actual_response = requests.post(url, json=bike_data).json()

expected_response = {'count': 2475.518697884526}

diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

assert 'type_changes' not in diff
assert 'values_changed' not in diff