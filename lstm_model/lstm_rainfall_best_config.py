# ===  ===
#   WandB run  sweep 
#  sweep  "threshold" 
#  

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
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
from sklearn.metrics import accuracy_score
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


import os
os.makedirs("result", exist_ok=True)

# ========== Data Preprocessing ==========
def load_and_preprocess():
    df = pd.read_csv("weather_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date", "RainToday", "RainTomorrow"], inplace=True)

    def wind_direction_to_angle(direction):
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        return directions.index(direction) * (360 / len(directions)) if direction in directions else np.nan

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].map(lambda x: wind_direction_to_angle(x) if isinstance(x, str) else 0)
        df[col + "_sin"] = np.sin(np.radians(df[col]))
        df[col + "_cos"] = np.cos(np.radians(df[col]))

    df["ThermalStress"] = df["MaxTemp"] * df["Humidity3pm"] / 100
    df["WindEnergy"] = df["WindGustSpeed"] ** 2 * df["WindSpeed3pm"]
    df["PressureGradient"] = abs(df["Pressure3pm"] - df["Pressure9am"])
    df["WindSynergy"] = (
        df["WindGustDir_sin"] * df["WindDir9am_sin"] +
        df["WindGustDir_cos"] * df["WindDir9am_cos"] +
        df["WindDir3pm_sin"] * df["WindGustDir_sin"] +
        df["WindDir3pm_cos"] * df["WindGustDir_cos"]
    )
    df["SeasonalFactor"] = np.sin(2 * np.pi * df["Month"] / 12)

    high_corr_features = ["MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm",
                          "WindDir9am_sin", "WindDir3pm_sin", "WindDir9am_cos", "WindDir3pm_cos",
                          "Temp9am", "Temp3pm", "Pressure9am", "Cloud9am", "Year", "Month", "Day"]
    df.drop(columns=high_corr_features, inplace=True)

    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(float)

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


class Config:
    batch_size = 128
    dropout1 = 0.2
    dropout2 = 0.4
    l2_reg = 5e-05
    learning_rate = 0.0006561964801432853
    lstm_units1 = 128
    lstm_units2 = 32
    threshold = 0.4803189173327605

# ========== Training Function ==========
def train():
    config = Config()

    #  run  sweep 
    
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

    # Predict probabilities
    y_prob = model.predict(X_test).flatten()
    y_pred = (y_prob >= config.threshold).astype(int)

    # Print evaluation metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Rain", "Rain"]))
    print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
    print("Accuracy Score:", accuracy_score(y_test, y_pred))

    # ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = {:.2f})'.format(roc_auc_score(y_test, y_prob)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig("result/roc_curve.png")
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig("result/confusion_matrix.png")
    plt.close()

    y_pred = (model.predict(X_test) >= config.threshold).astype(int).flatten()

    

# ========== Main Entry ==========
train()