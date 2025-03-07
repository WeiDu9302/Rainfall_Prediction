
import torch
from data_processor import DataProcessor
from model import StockPredictor


def predict_latest(horizon=5):
    processor = DataProcessor()
    sequences = processor.create_features()
    seq = sequences[-1][0]  # 最新窗口

    model = StockPredictor(input_size=seq.shape[1])
    model.load_state_dict(torch.load('stock_model.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        current_seq = torch.FloatTensor(seq).unsqueeze(0)
        for _ in range(horizon):
            pred = model(current_seq)
            predictions.append(pred.item())

            next_step = current_seq.squeeze(0)[1:].clone()
            next_features = current_seq.squeeze(0)[-1].clone()
            next_features[-1] = pred
            next_step = torch.vstack([next_step, next_features])
            current_seq = next_step.unsqueeze(0)

    # 反标准化 Close 值
    close_mean = processor.scaler.mean_[processor.close_index]
    close_std = processor.scaler.scale_[processor.close_index]
    real_prices = [close_mean + p * close_std for p in predictions]

    print(f"Predictions for next {horizon} day(s): {predictions}")
    print(f"Real predicted prices: {real_prices}")


if __name__ == '__main__':
    predict_latest(horizon=5)
