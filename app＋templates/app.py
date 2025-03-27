from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from predict import predict_rainfall_past_5_days, feature_list
import joblib

app = Flask(__name__)


scaler_X = joblib.load('scaler_X.save')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
 
    df = pd.DataFrame(data)
    

    probability = predict_rainfall_past_5_days(
        "rain_prediction_lstm.h5",
        df,
        scaler_X,
        feature_list
    )
    
    return jsonify({"probability": f"{probability:.2%}"})

if __name__ == '__main__':
    app.run(debug=True) 
