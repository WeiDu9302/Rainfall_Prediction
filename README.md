# Rainfall_Prediction Oracle – SNS Project 2024/2025

**Team Members:**
- Wei Du
- Yuning Zhou
- Haowen Yang

  
This project implements a time series forecasting Oracle chatbot for rainfall prediction using an LSTM model. It was developed for the UCL Software for Network Services (SNS) module.

The Oracle predicts future rainfall based on historical weather data and is accessed via a simple terminal-based client-server interface.

## Project Structure

```text
Rainfall_Prediction/
├── lstm_model/                         [ AUTHOR: Wei Du ]
│   ├── EDA.ipynb                      # Exploratory data analysis
│   ├── lstm_rainfall_best_config.py   # Final LSTM with best config
│   ├── lstm_rainfall_prediction_with_WandB.py  # Training with WandB tuning
│   ├── sweep_config.yaml              # WandB hyperparameter sweep setup
│   └── weather_data.csv               # Raw weather dataset
│
├── Example_Predict/                    [ AUTHOR: Wei Du ]
│   ├── predict.py                     # Script for loading model and predicting
│   ├── rain_prediction_lstm.h5        # Trained LSTM model
│   └── scaler_X.save                  # Scaler for input preprocessing
│
├── server-client/                      [ AUTHOR: Yuning Zhou ]
│   ├── client.py                      # Simple terminal client
│   └── server.py                      # Server hosting the Oracle model
│
├── user_interface/                     [ AUTHOR: Haowen Yang ]
│   ├── app.py                         # Flask app
│   └── index.html                     # Web interface template
│
└── README.md                           # Project documentation
```


## Features

- Time series forecasting using LSTM
- Model tuning using Weights & Biases (WandB)
- Command-line based Oracle chatbot (client-server architecture)
- Optional web interface built with Flask

## How to Run

This section describes how to train the model, run predictions, and interact with the Oracle via both terminal and web interfaces.

### 1. Train the model

Navigate to the `lstm_model` folder and run:

```bash
cd lstm_model
python lstm_rainfall_prediction_with_WandB.py
```

This script performs:
- Data preprocessing and feature engineering
- LSTM model training
- Hyperparameter tuning using Weights & Biases (WandB)
- Saving the trained model (`rain_prediction_lstm.h5`) and scaler (`scaler_X.save`)

---

### 2. Predict using the saved model

Navigate to the `Example_Predict` folder and run:

```bash
cd Example_Predict
python predict.py
```

This script loads the saved model and scaler, accepts user input, and makes a rainfall prediction.

---

### 3. Start the Oracle chatbot (terminal client-server interface)

Open **two separate terminals**.

**Terminal 1: Start the server**

```bash
cd server-client
python server.py
```

**Terminal 2: Start the client**

```bash
cd server-client
python client.py
```

You will see a prompt like:

```
Hello, I’m the Oracle. How can I help you today?
```

Type supported sentences like:

```
Can you give me the rainfall prediction for tomorrow?
```

The server will respond with a forecast based on the LSTM model.

<img width="576" alt="1" src="https://github.com/user-attachments/assets/04ba3eb5-5de3-4b24-bf96-953d24f07f0e" />


In this screenshot, the user uses predefined weather data for five consecutive days. The chatbot performs validation, applies default values where necessary, and confirms the input before sending it to the server. The predicted probability of rain for the next day is then displayed in a clean and structured format.

<img width="576" alt="2" src="https://github.com/user-attachments/assets/5fb896ad-6f15-4ddf-9540-7acf4715a55d" />


This screenshot shows the server receiving a POST request from the client and printing the raw input data in tabular form. It then applies feature engineering and normalization before reshaping the data into the format required by the LSTM model. Finally, the model generates a prediction, which is sent back to the client.



---

### 4. Launch the web interface

Navigate to the `user_interface` folder and run:

```bash
cd user_interface
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Use the web form to submit queries to the Oracle and view predictions in your browser.

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


