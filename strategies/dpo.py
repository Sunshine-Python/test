import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

def calculate_dpo(close, period=20):
    shift = period // 2 + 1
    sma = close.rolling(window=period).mean()
    dpo = close.shift(shift) - sma
    return dpo

class DPOStrategy(Strategy):
    period = 20
    threshold = 0
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        close = pd.Series(self.data.Close)
        self.dpo = self.I(calculate_dpo, close, self.period)
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
            if self.dpo[-1] > self.threshold:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.dpo[-1] < -self.threshold and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'



def dpo_viz(data, period=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    dpo = calculate_dpo(data['Close'], period)

    data['Date'] = pd.to_datetime(data['Datetime']).dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('DPO Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, dpo, label='DPO', color='orange')
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('DPO')
    ax2.legend()
    ax2.grid(True)
    
    plt.xticks([data[data['Date'] == date].index[0] for date in daily_indices],
               [date.strftime('%Y-%m-%d') for date in daily_indices],
               rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)

def run_dpo(ticker, start_date, end_date, cash, commission, period, threshold, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    DPOStrategy.period = period
    DPOStrategy.threshold = threshold
    DPOStrategy.stop_loss_pct = stop_loss_pct
    DPOStrategy.take_profit_pct = take_profit_pct
    DPOStrategy.enable_shorting = enable_shorting
    DPOStrategy.enable_stop_loss = enable_stop_loss
    DPOStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(DPOStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None