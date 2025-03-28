from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import json

# ========== Initialize Flask App ==========
app = Flask(__name__)

# ========== Load the Scaler ==========
try:
    scaler = joblib.load("scaler_X.save")
    print("Scaler loaded successfully!")
except Exception as e:
    scaler = None
    print(f"Error loading scaler: {e}")

# ========== Load the Model ==========
MODEL_PATH = "rain_prediction_lstm.h5"
try:
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# ========== Feature List and Helper Functions ==========
feature_list = ['Evaporation', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                'Pressure3pm', 'Cloud3pm', 'WindGustDir_sin', 'WindGustDir_cos',
                'AvgTemp', 'AvgWindSpeed', 'WindDir_sin', 'WindDir_cos']

directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
              'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

def wind_direction_to_angle(direction):
    angle_per_direction = 360 / len(directions)
    return directions.index(direction) * angle_per_direction if direction in directions else 0

def preprocess_input(raw_df):
    df = raw_df.copy()

    # Encode wind direction to angle → sin/cos
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].map(lambda x: wind_direction_to_angle(x))
        df[col + "_sin"] = np.sin(np.radians(df[col]))
        df[col + "_cos"] = np.cos(np.radians(df[col]))
    df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm"], inplace=True)

    # Create averaged features
    df["AvgTemp"] = (df["MinTemp"] + df["MaxTemp"]) / 2
    df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
    df["WindDir_sin"] = (df["WindDir9am_sin"] + df["WindDir3pm_sin"]) / 2
    df["WindDir_cos"] = (df["WindDir9am_cos"] + df["WindDir3pm_cos"]) / 2

    # Drop unused columns
    drop_cols = ["MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm",
                 "WindDir9am_sin", "WindDir3pm_sin", "WindDir9am_cos", "WindDir3pm_cos",
                 "Temp9am", "Temp3pm", "Pressure9am", "Cloud9am", "Year", "Month", "Day",
                 "Rainfall", "RainToday", "RainTomorrow", "Date"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)

    df = df[feature_list]

    # Apply scaling
    scaled_data = scaler.transform(df)
    return scaled_data

# ========== Routes ==========
@app.route('/')
def home():
    return "Rainfall Prediction Server is Running! Use POST /predict."

@app.route('/predict', methods=['POST'])
def predict_rainfall():
    if model is None:
        return jsonify({"error": "Model not properly initialized."}), 500
    if scaler is None:
        return jsonify({"error": "Scaler not properly initialized."}), 500

    try:
        data = request.get_json()
        print("\n Received JSON data:")
        print(json.dumps(data, indent=2))

        if "weather_data" not in data:
            return jsonify({"error": "Missing 'weather_data' field"}), 400

        weather_data = data["weather_data"]
        df = pd.DataFrame(weather_data)
        print("\n Raw input DataFrame:")
        print(df.head())

        processed_data = preprocess_input(df)
        print(f"\n Processed data shape: {processed_data.shape}")
        print(processed_data)

        input_data = np.array([processed_data])
        print(f"\n Model input shape: {input_data.shape}")

        predicted_prob = model.predict(input_data)[0, 0]
        print(f"\n Predicted probability: {predicted_prob:.4f}")

        return jsonify({"rainfall_probability": float(predicted_prob)})
    except Exception as e:
        print(" Exception during prediction:", str(e))
        return jsonify({"error": str(e)}), 500

# ========== Run the Server ==========
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
