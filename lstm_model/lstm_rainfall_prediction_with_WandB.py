"""
Filename: lstm_rainfall_prediction_with_WandB.py.py

Description: LSTM-based rainfall prediction model with WandB sweep support and simplified feature engineering.

Author: Wei Du
Created: March 16, 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import yaml
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbCallback
from sklearn.metrics import accuracy_score
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# ========== Data Preprocessing ==========
def load_and_preprocess():
    # Load the dataset
    df = pd.read_csv("weather_data.csv")

    # Parse date and extract time features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)

    # Drop target leakage columns
    df.drop(columns=["RainToday", "RainTomorrow"], inplace=True)

    # Convert wind direction strings into angles (in degrees)
    def wind_direction_to_angle(direction):
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        return directions.index(direction) * (360 / len(directions)) if direction in directions else np.nan

    # Encode wind direction to sin/cos
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].map(lambda x: wind_direction_to_angle(x) if isinstance(x, str) else 0)
        df[col + "_sin"] = np.sin(np.radians(df[col]))
        df[col + "_cos"] = np.cos(np.radians(df[col]))

    # Drop original wind direction columns
    df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm"], inplace=True)

    # Merge related features into average representations
    df["AvgTemp"] = (df["MinTemp"] + df["MaxTemp"]) / 2
    df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
    df["WindDir_sin"] = (df["WindDir9am_sin"] + df["WindDir3pm_sin"]) / 2
    df["WindDir_cos"] = (df["WindDir9am_cos"] + df["WindDir3pm_cos"]) / 2

    # Drop redundant or highly correlated features
    high_corr_features = [
        "MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm",
        "WindDir9am_sin", "WindDir3pm_sin", "WindDir9am_cos", "WindDir3pm_cos",
        "Temp9am", "Temp3pm", "Pressure9am", "Cloud9am", "Year", "Month", "Day"
    ]
    df.drop(columns=high_corr_features, inplace=True)

    # Fill missing values
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(float)

    # Create binary rainfall target
    df["RainBinary"] = df["Rainfall"].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=["Rainfall"], inplace=True)

    return df


# ========== Data Preparation ==========
df = load_and_preprocess()
features, target = df.columns.drop('RainBinary'), 'RainBinary'
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
test_df, val_df = train_test_split(test_df, test_size=0.2, shuffle=False)

scaler_X = MinMaxScaler()
train_scaled = scaler_X.fit_transform(train_df[features])
test_scaled = scaler_X.transform(test_df[features])
val_scaled = scaler_X.transform(val_df[features])

def createXY(X, y, n_past=5):
    X_data, y_data = [], []
    for i in range(n_past, len(X)):
        X_data.append(X[i - n_past:i])
        y_data.append(y[i])
    return np.array(X_data), np.array(y_data)

train_y, test_y, val_y = train_df[target].values, test_df[target].values, val_df[target].values
X_train, y_train = createXY(train_scaled, train_y)
X_test, y_test = createXY(test_scaled, test_y)
X_val, y_val = createXY(val_scaled, val_y)

y_train = y_train.flatten().astype(int)
y_val = y_val.flatten().astype(int)
y_test = y_test.flatten().astype(int)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ========== Model Definition ==========
def build_model(input_shape, config):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(config.lstm_units1, return_sequences=True, kernel_regularizer=l2(config.l2_reg)))(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dropout(config.dropout1)(x)
    x = Bidirectional(LSTM(config.lstm_units2, kernel_regularizer=l2(config.l2_reg)))(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout2)(x)
    x = Dense(32, activation='swish', kernel_regularizer=l2(config.l2_reg))(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# ========== Training Function ==========
def train():
    wandb.init(project="LSTM-Rainfall-Prediction")
    config = wandb.config

    # 添加 run 命名：根据 sweep 的参数
    wandb.run.name = f"lr={config.learning_rate}_bs={config.batch_size}_units={config.lstm_units1}-{config.lstm_units2}_drop={config.dropout1}-{config.dropout2}_th={config.threshold}"

    model = build_model((X_train.shape[1], X_train.shape[2]), config)
    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        WandbCallback()
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=config.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        shuffle=True
    )

    model.load_weights('best_model.h5')
    y_pred = (model.predict(X_test) >= config.threshold).astype(int).flatten()

    wandb.log({
        "test_auc": roc_auc_score(y_test, y_pred),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test, preds=y_pred, class_names=["No Rain", "Rain"]
        )
    })

# ========== Main Entry ==========
if __name__ == "__main__":
    with open("sweep_config8.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project="LSTM-Rainfall-Prediction")
    wandb.agent(sweep_id, function=train, count=50)
