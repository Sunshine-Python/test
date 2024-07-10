import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

def calculate_stochastic(high, low, close, k_period, d_period):
    k = np.zeros_like(close)
    d = np.zeros_like(close)
    
    for i in range(k_period, len(close)):
        low_min = np.min(low[i-k_period+1:i+1])
        high_max = np.max(high[i-k_period+1:i+1])
        k[i] = 100 * (close[i] - low_min) / (high_max - low_min)
    
    d = np.convolve(k, np.ones(d_period)/d_period, mode='same')
    
    return k, d

class StochStrategy(Strategy):
    k_period = 14
    d_period = 3
    overbought = 80
    oversold = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.k, self.d = self.I(calculate_stochastic, self.data.High, self.data.Low, self.data.Close, 
                                self.k_period, self.d_period)
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
            if self.k[-1] < self.oversold and self.k[-1] > self.d[-1]:
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif self.k[-1] > self.overbought and self.k[-1] < self.d[-1] and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'

def stoch_viz(data, k_period=14, d_period=3, overbought=80, oversold=20):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    k, d = calculate_stochastic(data['High'], data['Low'], data['Close'], k_period, d_period)
    data['k'] = k
    data['d'] = d
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('Stochastic Oscillator Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, data['k'], label='%K', color='orange')
    ax2.plot(data.index, data['d'], label='%D', color='green')
    ax2.axhline(overbought, color='red', linestyle='--', label='Overbought')
    ax2.axhline(oversold, color='red', linestyle='--', label='Oversold')
    ax2.set_ylabel('Stochastic Oscillator')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-5, 105)  # Set y-axis limits for the Stochastic Oscillator
    
    plt.xticks(
        ticks=[data[data['Date'] == date].index[0] for date in daily_indices],
        labels=[date.strftime('%Y-%m-%d') for date in daily_indices],
        rotation=30
    )
    
    plt.tight_layout()
    st.pyplot(fig)

def run_stoch_strategy(ticker, start_date, end_date, cash, commission, k_period, d_period, overbought, oversold, 
                       stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    StochStrategy.k_period = k_period
    StochStrategy.d_period = d_period
    StochStrategy.overbought = overbought
    StochStrategy.oversold = oversold
    StochStrategy.stop_loss_pct = stop_loss_pct
    StochStrategy.take_profit_pct = take_profit_pct
    StochStrategy.enable_shorting = enable_shorting
    StochStrategy.enable_stop_loss = enable_stop_loss
    StochStrategy.enable_take_profit = enable_take_profit
    
    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(StochStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None