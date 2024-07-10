import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

def calculate_adx(high, low, close, period=14):
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        h_diff = high[i] - high[i-1]
        l_diff = low[i-1] - low[i]
        
        plus_dm[i] = max(h_diff, 0) if h_diff > l_diff else 0
        minus_dm[i] = max(l_diff, 0) if l_diff > h_diff else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    tr = pd.Series(tr).rolling(window=period).sum()
    plus_dm = pd.Series(plus_dm).rolling(window=period).sum()
    minus_dm = pd.Series(minus_dm).rolling(window=period).sum()
    
    plus_di = 100 * plus_dm / tr
    minus_di = 100 * minus_dm / tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = pd.Series(dx).rolling(window=period).mean()
    
    return plus_di, minus_di, adx

class ADXStrategy(Strategy):
    period = 14
    adx_threshold = 25
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.plus_di, self.minus_di, self.adx = self.I(calculate_adx, self.data.High, self.data.Low, self.data.Close, self.period)
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
            if self.adx[-1] > self.adx_threshold:
                if crossover(self.plus_di, self.minus_di):
                    self.buy()
                    self.entry_price = self.data.Close[-1]
                    self.position_type = 'long'
                elif crossover(self.minus_di, self.plus_di) and self.enable_shorting:
                    self.sell()
                    self.entry_price = self.data.Close[-1]
                    self.position_type = 'short'

def adx_viz(data, period=14, adx_threshold=25):
    data = data[data['Volume'] > 0].copy()
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    plus_di, minus_di, adx = calculate_adx(data['High'], data['Low'], data['Close'], period)
    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ax1.plot(data.index, data['Close'], label='Price', color='blue')
    ax1.set_title('ADX Strategy Visualization')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(data.index, plus_di, label='+DI', color='green')
    ax2.plot(data.index, minus_di, label='-DI', color='red')
    ax2.plot(data.index, adx, label='ADX', color='purple')
    ax2.axhline(y=adx_threshold, color='gray', linestyle='--', label=f'ADX Threshold ({adx_threshold})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('ADX / DI')
    ax2.legend()
    ax2.grid(True)
    
    # Set y-axis limits for the second subplot
    y_min = min(plus_di.min(), minus_di.min(), adx.min(), adx_threshold)
    y_max = max(plus_di.max(), minus_di.max(), adx.max(), adx_threshold)
    ax2.set_ylim(max(0, y_min - 5), min(100, y_max + 5))
    
    plt.xticks([data[data['Date'] == date].index[0] for date in daily_indices],
               [date.strftime('%Y-%m-%d') for date in daily_indices],
               rotation=30)
    
    plt.tight_layout()
    st.pyplot(fig)

def run_adx(ticker, start_date, end_date, cash, commission, period, adx_threshold, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    ADXStrategy.period = period
    ADXStrategy.adx_threshold = adx_threshold
    ADXStrategy.stop_loss_pct = stop_loss_pct
    ADXStrategy.take_profit_pct = take_profit_pct
    ADXStrategy.enable_shorting = enable_shorting
    ADXStrategy.enable_stop_loss = enable_stop_loss
    ADXStrategy.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(ADXStrategy, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None