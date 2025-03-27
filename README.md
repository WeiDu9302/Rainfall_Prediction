# Rainfall_Prediction Oracle – SNS Project 2024/2025

This project implements a time series forecasting Oracle chatbot for rainfall prediction using an LSTM model. It was developed for the UCL Software for Network Services (SNS) module.

The Oracle predicts future rainfall based on historical weather data and is accessed via a simple terminal-based client-server interface.

## Project Structure

```text
Rainfall_Prediction/
├── lstm_model/                        # Author: Wei Du
│   ├── EDA.ipynb                      # Exploratory data analysis
│   ├── lstm_rainfall_best_config.py  # Final LSTM with best config
│   ├── lstm_rainfall_prediction_with_WandB.py  # Training with WandB tuning
│   ├── sweep_config.yaml             # WandB hyperparameter sweep setup
│   └── weather_data.csv              # Raw weather dataset
│
├── Example_Predict/                  # Author: Wei Du
│   ├── predict.py                    # Script for loading model and predicting
│   ├── rain_prediction_lstm.h5       # Trained LSTM model
│   └── scaler_X.save                 # Scaler for input preprocessing
│
├── server-client/                    # Author: Yuning Zhou
│   ├── client.py                     # Simple terminal client
│   └── server.py                     # Server hosting the Oracle model
│
├── user_interface/                   # Author: Haowen Yang
│   ├── app.py                        # Flask app
│   └── index.html                    # Web interface template
│
└── README.md                         # Project documentation



## Features

- Time series forecasting using LSTM
- Model tuning using Weights & Biases (WandB)
- Command-line based Oracle chatbot (client-server architecture)
- Optional web interface built with Flask

## How to Run

### 1. Train the model

Navigate to the lstm_model folder and run:

### 2. Predict using saved model

Navigate to the Example_Predict folder and run:


### 3. Start the Oracle chatbot (client-server)

In one terminal (server side):


In another terminal (client side):


### 4. (Optional) Launch the web interface


Then open your browser and go to http://127.0.0.1:5000

## Requirements

- Python 3.x
- tensorflow or keras
- pandas
- numpy
- scikit-learn
- wandb (for hyperparameter tuning)
- flask (for web interface)

## License

This project is developed for educational purposes as part of the SNS coursework at UCL. Not intended for commercial use.


