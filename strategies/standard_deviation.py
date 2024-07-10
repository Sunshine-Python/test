import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

def rolling_std(array, n):
    result = np.full_like(array, np.nan)
    for i in range(n-1, len(array)):
        result[i] = np.std(array[i-n+1:i+1])
    return result

def rolling_mean(array, n):
    result = np.full_like(array, np.nan)
    for i in range(n-1, len(array)):
        result[i] = np.mean(array[i-n+1:i+1])
    return result

class StdDevStrategy(Strategy):
    period = 20
    multiplier = 2
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = self.data.Close
        self.sma = self.I(rolling_mean, close, self.period)
        self.std = self.I(rolling_std, close, self.period)
        self.upper = self.I(lambda: self.sma + self.multiplier * self.std)
        self.lower = self.I(lambda: self.sma - self.multiplier * self.std)
        self.entry_price = None
        self.position_type = None  # 'long' or 'short'

    def next(self):
        if self.position:
            if self.position_type == 'long':
                if self.enable_stop_loss and self.data.Close[-1] <= self.entry_price * (1 - self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and self.data.Close[-1] >= self.entry_price * (1 + self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None
            elif self.position_type == 'short':
                if self.enable_stop_loss and self.data.Close[-1] >= self.entry_price * (1 + self.stop_loss_pct / 100):
                    self.position.close()
                    self.position_type = None
                elif self.enable_take_profit and self.data.Close[-1] <= self.entry_price * (1 - self.take_profit_pct / 100):
                    self.position.close()
                    self.position_type = None

        if not self.position:
            if self.data.Close[-1] < self.lower[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.data.Close[-1] > self.upper[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def std_dev_viz(data, period=20, multiplier=2):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    sma = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper = sma + multiplier * std
    lower = sma - multiplier * std

    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.plot(data.index, upper, label='Upper Band', color='red', linestyle='--')
    ax1.plot(data.index, lower, label='Lower Band', color='green', linestyle='--')
    ax1.set_title('Standard Deviation Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, std, label='Standard Deviation', color='purple')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Standard Deviation')
    ax2.legend()
    ax2.grid(True)
    
    plt.xticks([data[data['Date'] == date].index[0] for date in daily_indices],
               [date.strftime('%Y-%m-%d') for date in daily_indices],
               rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)

def run_std_dev_strategy(ticker, start_date, end_date, cash, commission, period, multiplier, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    StdDevStrategy.period = period
    StdDevStrategy.multiplier = multiplier
    StdDevStrategy.stop_loss_pct = stop_loss_pct
    StdDevStrategy.take_profit_pct = take_profit_pct
    StdDevStrategy.enable_shorting = enable_shorting
    StdDevStrategy.enable_stop_loss = enable_stop_loss
    StdDevStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(StdDevStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None