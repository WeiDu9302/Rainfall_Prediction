import yfinance as yf
import pandas as pd
from fredapi import Fred
import numpy as np
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('FRED_API_KEY')


class DataProcessor:
    def __init__(self, ticker='AAPL'):
        self.ticker = ticker
        self.fred = Fred(api_key=API_KEY)

    def get_stock_data(self, days=365 * 5):
        # 让 yfinance 返回单层列
        df = yf.download(
            tickers=self.ticker,
            period=f'{days}d',
            auto_adjust=False,  # 显式关闭自动复权
            group_by="column"  # 不分组，返回单层列
        )

        if isinstance(df.columns, pd.MultiIndex):
            # 如果只下载了一个ticker，那第二层就是 [AAPL, AAPL, ...]
            # 通过 xs() 取出这个 ticker 对应的数据
            try:
                df = df.xs(key=self.ticker, level=1, axis=1)
            except Exception:
                pass  # 如果 xs() 失败，可能需要进一步调试

        # 现在应该是单层列了，选取需要的列
        # （有些情况列名开头会是大写；核实一下 df.columns)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

        # 强制使用 DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df

    def get_economic_data(self):
        cpi = self.fred.get_series('CPIAUCSL')
        unrate = self.fred.get_series('UNRATE')
        gdp = self.fred.get_series('GDP')
        fedfunds = self.fred.get_series('FEDFUNDS')
        indpro = self.fred.get_series('INDPRO')

        econ_df = pd.concat([cpi, unrate, gdp, fedfunds, indpro], axis=1)
        econ_df.columns = ['CPI', 'Unemployment', 'GDP', 'FEDFUNDS', 'INDPRO']
        econ_df.index = pd.to_datetime(econ_df.index)
        econ_df.sort_index(inplace=True)
        econ_df.fillna(method='ffill', inplace=True)
        econ_df.fillna(method='bfill', inplace=True)
        return econ_df

    def create_features(self, window_size=10):
        # 获取数据
        stock_df = self.get_stock_data()
        econ_df = self.get_economic_data()

        '''
        print("\n[DEBUG] Stock Data (head & index):")
        print(stock_df.head())
        print(stock_df.index)
        print("\n[DEBUG] Economic Data (head & index):")
        print(econ_df.head())
        print(econ_df.index)
        '''

        # asof 合并
        merged = pd.merge_asof(
            stock_df, econ_df,
            left_index=True, right_index=True,
            direction='backward'
        )
        merged.fillna(method='ffill', inplace=True)

        if 'Close' not in merged.columns:
            raise KeyError("'Close' column missing after merging. Check stock and economic data alignment.")

        #时间特征
        merged['Month'] = merged.index.month
        merged['DayOfWeek'] = merged.index.dayofweek

        #技术指标
        merged['RSI'] = self._calculate_rsi(merged['Close'])
        merged['MA5'] = merged['Close'].rolling(5).mean()

        # MACD
        ema12 = merged['Close'].ewm(span=12, adjust=False).mean()
        ema26 = merged['Close'].ewm(span=26, adjust=False).mean()
        merged['MACD'] = ema12 - ema26

        # Bollinger Bands
        merged['BB_high'] = merged['Close'].rolling(20).mean() + 2 * merged['Close'].rolling(20).std()
        merged['BB_low'] = merged['Close'].rolling(20).mean() - 2 * merged['Close'].rolling(20).std()

        # ATR
        high_low = merged['High'] - merged['Low']
        high_close = (merged['High'] - merged['Close'].shift()).abs()
        low_close = (merged['Low'] - merged['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        merged['ATR'] = tr.rolling(14).mean()

        # 6. 异常值过滤
        z_scores = (merged - merged.mean()) / merged.std()
        merged = merged[(z_scores.abs() <= 3).all(axis=1)]
        if merged.empty:
            raise ValueError("Merged data is empty after outlier removal.")

        # 7. 特征归一化 （排除月份、星期几这种分类特征）
        scaler = StandardScaler()
        features_to_scale = merged.drop(columns=['Month', 'DayOfWeek']).columns
        merged[features_to_scale] = scaler.fit_transform(merged[features_to_scale])

        # 8. 滑动窗口
        sequences = []
        for i in range(window_size, len(merged)):
            seq = merged.iloc[i - window_size:i]
            target = merged['Close'].iloc[i]
            sequences.append((seq.values, target))

        return np.array(sequences)

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
