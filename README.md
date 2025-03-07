
# 📊 Stock Price Prediction with LSTM

基于股票历史数据、宏观经济数据和技术指标的时间序列预测项目，使用 LSTM 神经网络对未来股票收盘价进行预测。

---

## ✅ 项目结构

```
├── data_processor.py    # 数据处理与特征工程
├── model.py             # LSTM 模型训练
├── predict.py           # 模型预测
├── .env                 # 存放 FRED API Key
├── stock_model.pth      # 训练好的模型参数（训练后生成）
├── requirements.txt     # 环境依赖
```

---

## ✅ 环境依赖

安装依赖包：

```bash
pip install -r requirements.txt
```

**requirements.txt 示例**：

```
torch
yfinance
pandas
numpy
sklearn
python-dotenv
fredapi
```

---

## ✅ 数据来源

- **股票数据**：使用 `yfinance` 获取指定股票（默认 `AAPL`）过去 5 年的历史数据。
- **宏观经济数据**：使用 FRED API 获取 CPI、失业率、GDP、联邦基金利率、工业生产指数。

**注意：** 需要在 `.env` 文件中配置 FRED API Key，例如：

```
FRED_API_KEY=你的FRED_API_KEY
```

API Key 免费申请：https://fred.stlouisfed.org/

---

## ✅ 特征工程

处理步骤：
- 合并股票数据与宏观经济数据。
- 生成时间特征（月份、星期几）。
- 计算技术指标：
  - RSI（相对强弱指标）
  - MA5（5日均线）
  - MACD（指数平滑异同移动平均线）
  - Bollinger Bands（布林带上下轨）
  - ATR（平均真实波幅）
- 异常值处理（Z-score 过滤）。
- 标准化特征（除月份与星期几）。

最终使用滑动窗口构建时间序列数据集。

---

## ✅ 模型设计

采用 `LSTM` 模型结构：
- 输入：动态匹配特征维度。
- 隐藏层大小：64
- LSTM 层数：2
- Dropout：0.3
- 输出：下一个时间步的收盘价。

---

## ✅ 训练模型

运行以下命令训练模型并保存权重：

```bash
python -c "from model import train_model; train_model()"
```

训练完成后会生成 `stock_model.pth` 文件。

---

## ✅ 预测示例

使用最新数据进行预测：

```bash
python predict.py
```

默认预测未来 5 天的收盘价。预测逻辑采用滚动窗口，将前一步的预测加入下一步输入。

---

## ✅ 可扩展优化方向

- 增强特征工程（更多技术指标、统计特征）。
- 优化模型（GRU、Transformer、Attention 机制）。
- 引入超参数调优（学习率、窗口大小、隐藏层大小等）。
- 改进预测接口，支持用户输入指定预测周期（如“明天”“下周”）。

---

## ✅ 注意事项

- 运行前请确保 `.env` 配置正确。
- 初次运行前需先训练模型，确保 `stock_model.pth` 存在。
- 数据更新频率取决于 yfinance 和 FRED 的数据源。
