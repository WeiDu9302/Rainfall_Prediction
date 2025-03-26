import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# 特征列表（必须与训练时一致）
feature_list = ['Evaporation', 'Sunshine', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm',
                'Pressure3pm', 'Cloud3pm', 'WindGustDir_sin', 'WindGustDir_cos',
                'AvgTemp', 'AvgWindSpeed', 'WindDir_sin', 'WindDir_cos']

# 风向转角度函数
def wind_direction_to_angle(direction):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    angle_per_direction = 360 / len(directions)
    return directions.index(direction) * angle_per_direction if direction in directions else 0

# 数据预处理函数（与训练阶段一致）
def preprocess_input(raw_df):
    df = raw_df.copy()

    # 风向数据sin/cos编码
    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        df[col] = df[col].map(lambda x: wind_direction_to_angle(x))
        df[col + "_sin"] = np.sin(np.radians(df[col]))
        df[col + "_cos"] = np.cos(np.radians(df[col]))
    df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm"], inplace=True)

    # 合并特征
    df["AvgTemp"] = (df["MinTemp"] + df["MaxTemp"]) / 2
    df["AvgWindSpeed"] = (df["WindSpeed9am"] + df["WindSpeed3pm"]) / 2
    df["WindDir_sin"] = (df["WindDir9am_sin"] + df["WindDir3pm_sin"]) / 2
    df["WindDir_cos"] = (df["WindDir9am_cos"] + df["WindDir3pm_cos"]) / 2

    # 删除冗余列
    drop_cols = ["MinTemp", "MaxTemp", "WindSpeed9am", "WindSpeed3pm",
                 "WindDir9am_sin", "WindDir3pm_sin", "WindDir9am_cos", "WindDir3pm_cos",
                 "Temp9am", "Temp3pm", "Pressure9am", "Cloud9am", "Year", "Month", "Day",
                 "Rainfall", "RainToday", "RainTomorrow", "Date"]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # 缺失值处理
    df.fillna(df.mean(), inplace=True)
    df.fillna(0, inplace=True)

    return df[feature_list]

# 预测函数
def predict_rainfall_past_5_days(model_path, past_5_days_raw_data, scaler, feature_list):
    model = load_model(model_path)
    processed_data = preprocess_input(past_5_days_raw_data)
    scaled_data = scaler.transform(processed_data)
    input_data = np.array([scaled_data])
    predicted_prob = model.predict(input_data)[0, 0]
    return predicted_prob

# ================== 独立运行示例 ==================

# 加载保存的scaler对象（确保路径正确）
scaler_X = joblib.load('scaler_X.save')

# 加载测试数据示例（真实输入时替换此处即可）
example_raw_data = pd.DataFrame({
    "MinTemp": [19.5, 19.5, 21.6, 20.2, 19.7],
    "MaxTemp": [22.4, 25.6, 24.5, 22.8, 25.7],
    "Rainfall": [15.6, 6, 6.6, 18.8, 77.4],
    "Evaporation": [6.2, 3.4, 2.4, 2.2, 4.8],
    "Sunshine": [0, 2.7, 0.1, 0, 0],
    "WindGustDir": ["W", "W", "W", "W", "W"],
    "WindGustSpeed": [41, 41, 41, 41, 41],
    "WindDir9am": ["S", "W", "ESE", "NNE", "NNE"],
    "WindDir3pm": ["SSW", "E", "ESE", "E", "W"],
    "WindSpeed9am": [17, 9, 17, 22, 11],
    "WindSpeed3pm": [20, 13, 2, 20, 6],
    "Humidity9am": [92, 83, 88, 83, 88],
    "Humidity3pm": [84, 73, 86, 90, 74],
    "Pressure9am": [1017.6, 1017.9, 1016.7, 1014.2, 1008.3],
    "Pressure3pm": [1017.4, 1016.4, 1015.6, 1011.8, 1004.8],
    "Cloud9am": [8, 7, 7, 8, 8],
    "Cloud3pm": [8, 7, 8, 8, 8]
})

# 执行预测
rain_probability = predict_rainfall_past_5_days(
    "rain_prediction_lstm.h5",
    example_raw_data,
    scaler_X,
    feature_list
)

print(f"Predicted rainfall probability for tomorrow: {rain_probability:.2%}")
