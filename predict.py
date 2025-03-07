import torch
from data_processor import DataProcessor
from model import StockPredictor

def predict_latest(horizon=1):
    processor = DataProcessor()
    seq = processor.create_features()[-1][0]  # 最新窗口
    model = StockPredictor(input_size=seq.shape[1])
    model.load_state_dict(torch.load('stock_model.pth'))
    model.eval()

    predictions = []
    with torch.no_grad():
        current_seq = torch.FloatTensor(seq).unsqueeze(0)
        for _ in range(horizon):
            pred = model(current_seq)
            predictions.append(pred.item())

            # 更新序列，把最新预测值加入，移除最早的一步
            next_step = current_seq.squeeze(0)[1:].clone()
            # 假设 Close 是最后一列特征
            next_features = current_seq.squeeze(0)[-1].clone()
            next_features[-1] = pred  # 替换 Close 特征
            next_step = torch.vstack([next_step, next_features])
            current_seq = next_step.unsqueeze(0)

    print(f"Predictions for next {horizon} day(s): {predictions}")


if __name__ == '__main__':
    # 示例：预测未来5天
    predict_latest(horizon=5)

