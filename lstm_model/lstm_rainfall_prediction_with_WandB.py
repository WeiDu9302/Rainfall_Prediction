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

# ========== Data Preprocessing ==========
def load_and_preprocess():
    df = pd.read_csv("weather_data.csv")

    # Extract date-related features
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df.drop(columns=["Date"], inplace=True)

    # Remove columns that contain future information
    df.drop(columns=['RainToday', 'RainTomorrow'], inplace=True)

    # Convert wind direction categories to angles, then use sine and cosine encoding
    def wind_direction_to_angle(direction):
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        return directions.index(direction) * (360 / len(directions)) if direction in directions else np.nan

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].map(lambda x: wind_direction_to_angle(x) if isinstance(x, str) else 0)
        df[col + "_sin"] = np.sin(np.radians(df[col]))
        df[col + "_cos"] = np.cos(np.radians(df[col]))

    # Create engineered features
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

    # Drop redundant or highly correlated features
    high_corr_features = ["MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm",
                          "WindDir9am_sin", "WindDir3pm_sin", "WindDir9am_cos", "WindDir3pm_cos",
                          "Temp9am", "Temp3pm", "Pressure9am", "Cloud9am", "Year", "Month", "Day"]
    df.drop(columns=high_corr_features, inplace=True)

    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)
    df = df.astype(float)

    # Create binary classification label
    df["RainBinary"] = df["Rainfall"].apply(lambda x: 1 if x > 0 else 0)
    df.drop(columns=["Rainfall"], inplace=True)

    return df

# ========== Data Preparation ==========
df = load_and_preprocess()
features, target = df.columns.drop('RainBinary'), 'RainBinary'
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
test_df, val_df = train_test_split(test_df, test_size=0.2, shuffle=False)

# Feature normalization
scaler_X = MinMaxScaler()
train_scaled = scaler_X.fit_transform(train_df[features])
test_scaled = scaler_X.transform(test_df[features])
val_scaled = scaler_X.transform(val_df[features])

# Function to build time-series window inputs
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

# Ensure label arrays are 1D integers
y_train = y_train.flatten().astype(int)
y_val = y_val.flatten().astype(int)
y_test = y_test.flatten().astype(int)

# Compute class weights to address imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print("Class weights:", class_weight_dict)

# ========== Model Definition ==========
def build_model(input_shape, config):
    inputs = Input(shape=input_shape)

    # First Bidirectional LSTM + Attention block
    x = Bidirectional(LSTM(config.lstm_units1, return_sequences=True, kernel_regularizer=l2(config.l2_reg)))(inputs)
    x = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = LayerNormalization()(x)
    x = Dropout(config.dropout1)(x)

    # Second Bidirectional LSTM block
    x = Bidirectional(LSTM(config.lstm_units2, kernel_regularizer=l2(config.l2_reg)))(x)
    x = BatchNormalization()(x)
    x = Dropout(config.dropout2)(x)

    # Fully connected layers
    x = Dense(32, activation='swish', kernel_regularizer=l2(config.l2_reg))(x)
    outputs = Dense(1, activation='sigmoid')(x)

    return Model(inputs, outputs)

# ========== Training Function ==========
def train():
    wandb.init(project="LSTM-Rainfall-Prediction")
    config = wandb.config

    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]), config)
    model.compile(
        optimizer=Adam(config.learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy']
    )

    # Define training callbacks
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        WandbCallback()
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        shuffle=True
    )

    # Evaluate best model on test set
    model.load_weights('best_model.h5')
    y_pred = (model.predict(X_test) >= config.threshold).astype(int).flatten()

    # Log results to WandB
    wandb.log({
        "test_auc": roc_auc_score(y_test, y_pred),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_test, preds=y_pred,
            class_names=["No Rain", "Rain"]
        )
    })

# ========== Main Entry ==========
if __name__ == "__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep_config, project="LSTM-Rainfall-Prediction")
    wandb.agent(sweep_id, function=train, count=50)
