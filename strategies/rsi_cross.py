import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

class RsiCross(Strategy):
    n1 = 10
    n2 = 20
    rsi_period = 14
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        # Calculate RSI
        close = self.data.Close
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Fill NaN values with 50 (neutral)
        
        self.rsi = self.I(lambda: rsi)
        self.rsi1 = self.I(SMA, self.rsi, self.n1)
        self.rsi2 = self.I(SMA, self.rsi, self.n2)
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
            if crossover(self.rsi1, self.rsi2):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.rsi2, self.rsi1) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def rsi_cross_viz(data, n1=10, n2=20, rsi_period=14):
    data = data[data['Volume'] > 0]    
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index

    # Calculate RSI
    close = data['Close']
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Fill NaN values with 50 (neutral)
    short_rsi = rsi.rolling(window=n1, min_periods=1).mean()
    long_rsi = rsi.rolling(window=n2, min_periods=1).mean()

    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(data.index, rsi, label='RSI', color='purple')
    ax2.plot(data.index, short_rsi, label=f'RSI SMA({n1})', color='orange')
    ax2.plot(data.index, long_rsi, label=f'RSI SMA({n2})', color='green')
    ax2.set_ylabel('RSI')
    ax2.set_ylim(-5, 105)
    ax2.legend()
    ax2.grid(True)

    plt.title('RSI Cross Visualization')
    plt.xlabel('Time')
    
    ax1.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30)

    plt.tight_layout()
    st.pyplot(fig)

def run_rsi_cross(ticker, start_date, end_date, cash, commission, n1, n2, rsi_period, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    RsiCross.n1 = n1
    RsiCross.n2 = n2
    RsiCross.rsi_period = rsi_period
    RsiCross.stop_loss_pct = stop_loss_pct
    RsiCross.take_profit_pct = take_profit_pct
    RsiCross.enable_shorting = enable_shorting
    RsiCross.enable_stop_loss = enable_stop_loss
    RsiCross.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(RsiCross, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None