import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from data_processor import DataProcessor
import numpy as np

class StockPredictor(nn.Module):
    def __init__(self, input_size=7, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


def train_model():
    processor = DataProcessor()
    sequences = processor.create_features()

    X = [s[0] for s in sequences]
    y = [s[1] for s in sequences]

    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 自动适配输入特征维度
    input_size = X_train.shape[2]
    model = StockPredictor(input_size=input_size)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            test_preds = model(X_test)
            mape = torch.mean(torch.abs((test_preds.squeeze() - y_test) / y_test)) * 100
        print(f"Epoch {epoch + 1} | Loss: {loss.item():.4f} | Test MAPE: {mape:.2f}%")

    torch.save(model.state_dict(), 'stock_model.pth')
