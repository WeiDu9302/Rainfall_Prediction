import requests
import json
import re

# Define the server URL
SERVER_URL = "http://127.0.0.1:5000/predict"

# Default values for missing inputs
defaults = {
    "MinTemp": 20.0,
    "MaxTemp": 28.0,
    "Evaporation": 5.0,
    "Sunshine": 7.0,
    "WindGustDir": "W",
    "WindGustSpeed": 35.0,
    "WindDir9am": "NW",
    "WindDir3pm": "E",
    "WindSpeed9am": 15.0,
    "WindSpeed3pm": 17.0,
    "Humidity9am": 75.0,
    "Humidity3pm": 70.0,
    "Pressure3pm": 1012.0,
    "Cloud3pm": 6.0
}

# Valid compass directions
directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# Value ranges for validation (min, max)
value_ranges = {
    "MinTemp": (-50, 60),
    "MaxTemp": (-50, 60),
    "Evaporation": (0, 50),
    "Sunshine": (0, 24),
    "WindGustSpeed": (0, 200),
    "WindSpeed9am": (0, 150),
    "WindSpeed3pm": (0, 150),
    "Humidity9am": (0, 100),
    "Humidity3pm": (0, 100),
    "Pressure3pm": (900, 1100),
    "Cloud3pm": (0, 9)
}

def get_direction(prompt, default):
    while True:
        value = input(f"{prompt} (e.g., N, S, E, W; default: {default}): ").strip().upper()
        if value == "":
            return default
        if value in directions:
            return value
        print("Invalid direction. Please try again.")

def get_float(prompt, default, key):
    while True:
        value = input(f"{prompt} (default: {default}): ").strip()
        if value == "":
            return default
        try:
            fval = float(value)
            if key in value_ranges:
                min_val, max_val = value_ranges[key]
                if not (min_val <= fval <= max_val):
                    confirm = input(f"{key} value {fval} is outside the normal range ({min_val}-{max_val}). Keep it? (y/n): ").strip().lower()
                    if confirm in ["y", "yes"]:
                        return fval
                    else:
                        continue
            return fval
        except ValueError:
            print("Invalid input! Please enter a numeric value.")

def parse_line_input(line):
    # Accepts comma, space, or semicolon separated input
    tokens = re.split(r'[\s,;]+', line.strip())
    return tokens

def collect_input():
    all_days = []
    print("\nPlease enter weather data for the past 5 days.")
    print("You can input one parameter at a time, or all at once separated by space/comma/semicolon.")
    print("Leave blank to use default values. Type 'quit' anytime to exit.")

    keys = list(defaults.keys())
    for day in range(1, 6):
        print(f"\nDay {day}:")
        row = {}
        for key in keys:
            if "Dir" in key:
                row[key] = get_direction(f"Enter {key}", defaults[key])
            else:
                row[key] = get_float(f"Enter {key}", defaults[key], key)
        all_days.append(row)

    print("\nYou have entered the following data:")
    for idx, day_data in enumerate(all_days):
        print(f"Day {idx+1}: {json.dumps(day_data)}")

    confirm = input("\nIs this data correct? (yes/no): ").strip().lower()
    if confirm not in ["yes", "y"]:
        return collect_input()
    return all_days

def send_request(data):
    print("\nSending data to the server...")
    try:
        response = requests.post(SERVER_URL, json={"weather_data": data})
        if response.status_code == 200:
            prob = response.json()['rainfall_probability']
            print("\nRainfall Prediction Result")
            print("-----------------------------")
            print(f"Probability of Rain Tomorrow: {prob:.2%}")
        else:
            print(f"\nServer returned error: {response.status_code}")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    print("==============================")
    print("Welcome to Rainfall Oracle")
    print("==============================")
    weather = collect_input()
    send_request(weather)
