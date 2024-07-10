import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from backtesting import Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from data_handler import fetch_data
from utils import run_backtest, plot_strat_perf, display_metrics

class SmaCross(Strategy):
    n1 = 10
    n2 = 20
    stop_loss_pct = 2.0
    take_profit_pct = 5.0
    enable_shorting = True
    enable_stop_loss = True
    enable_take_profit = True

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
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
            if crossover(self.sma1, self.sma2):
                self.buy()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'long'
            elif crossover(self.sma2, self.sma1) and self.enable_shorting:
                self.sell()
                self.entry_price = self.data.Close[-1]
                self.position_type = 'short'


## ORIGINAL SMA_CROSS VIZ 

def sma_cross_viz(data, n1=10, n2=20):
    # Modified to work with Streamlit
    data = data[data['Volume'] > 0]    
    data.reset_index(inplace=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    short_sma = np.convolve(data['Close'], np.ones(n1)/n1, mode='valid')
    long_sma = np.convolve(data['Close'], np.ones(n2)/n2, mode='valid')

    start_idx = max(n1, n2) - 1

    data['Date'] = data['Datetime'].dt.date
    daily_indices = data.groupby('Date').first().index
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(data.index[start_idx:], data['Close'][start_idx:], label='Price', color='blue')
    ax.plot(data.index[start_idx:], short_sma[start_idx - n1 + 1:], label=f'SMA({n1})', color='orange')
    ax.plot(data.index[start_idx:], long_sma[start_idx - n2 + 1:], label=f'SMA({n2})', color='green')

    ax.set_title('SMA Cross Visualization')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_xticks([data[data['Date'] == date].index[0] for date in daily_indices])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in daily_indices], rotation=30)
    
    ax.legend()
    ax.grid(True)
    
    st.pyplot(fig)


## SMA_CROSS_VIZ_PLT

"""
from matplotlib.animation import FuncAnimation

def sma_cross_viz(data, n1=10, n2=20):
    data = data[data['Volume'] > 0]
    data.reset_index(inplace=True, drop=True)
    
    if 'Datetime' not in data.columns:
        data['Datetime'] = data.index
    
    short_sma = np.convolve(data['Close'], np.ones(n1)/n1, mode='valid')
    long_sma = np.convolve(data['Close'], np.ones(n2)/n2, mode='valid')
    price = data['Close'].values

    fig, ax = plt.subplots(figsize=(14, 7))
    line_price, = ax.plot([], [], label='Price', color='blue')
    line_short_sma, = ax.plot([], [], label=f'SMA({n1})', color='orange')
    line_long_sma, = ax.plot([], [], label=f'SMA({n2})', color='green')

    def init():
        ax.set_xlim(0, len(price))
        ax.set_ylim(min(price) * 0.95, max(price) * 1.05)
        return line_price, line_short_sma, line_long_sma

    def update(frame):
        max_idx = min(frame, len(price))
        line_price.set_data(range(max_idx), price[:max_idx])
        if frame > n1:
            line_short_sma.set_data(range(n1-1, max_idx), short_sma[:max_idx-n1+1])
        if frame > n2:
            line_long_sma.set_data(range(n2-1, max_idx), long_sma[:max_idx-n2+1])
        return line_price, line_short_sma, line_long_sma

    ani = FuncAnimation(fig, update, frames=np.arange(1, len(price) + 1), init_func=init, blit=True)
    plt.legend()
    plt.show()
"""



## SMA_CROSS_VIZ_PLOTLY
"""
import plotly.graph_objs as go
import pandas as pd
import numpy as np

def sma_cross_viz(data, n1=10, n2=20):
    data = data[data['Volume'] > 0]
    data.reset_index(inplace=True, drop=True)
    
    short_sma = np.convolve(data['Close'], np.ones(n1)/n1, mode='valid')
    long_sma = np.convolve(data['Close'], np.ones(n2)/n2, mode='valid')
    price = data['Close'].values

    fig = go.Figure()

    # Add traces for price, short SMA, and long SMA
    fig.add_trace(go.Scatter(x=data.index, y=price, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=data.index[n1-1:], y=short_sma, mode='lines', name=f'SMA({n1})'))
    fig.add_trace(go.Scatter(x=data.index[n2-1:], y=long_sma, mode='lines', name=f'SMA({n2})'))

    # Add frames for the animation
    frames = [go.Frame(data=[go.Scatter(x=data.index[:k], y=price[:k]),
                             go.Scatter(x=data.index[n1-1:k], y=short_sma[:k-n1+1] if k > n1 else []),
                             go.Scatter(x=data.index[n2-1:k], y=long_sma[:k-n2+1] if k > n2 else [])],
                     layout=go.Layout(title_text=f"Frame {k}")) for k in range(1, len(data))]

    fig.frames = frames
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False,
                                        y=0,
                                        x=1.05,
                                        xanchor='right',
                                        yanchor='top',
                                        pad=dict(t=0, r=10),
                                        buttons=[dict(label='Play',
                                                      method='animate',
                                                      args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True, mode='immediate')])])])
    return fig
"""





def run_sma_cross(ticker, start_date, end_date, cash, commission, n1, n2, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit):
    SmaCross.n1 = n1
    SmaCross.n2 = n2
    SmaCross.stop_loss_pct = stop_loss_pct
    SmaCross.take_profit_pct = take_profit_pct
    SmaCross.enable_shorting = enable_shorting
    SmaCross.enable_stop_loss = enable_stop_loss
    SmaCross.enable_take_profit = enable_take_profit

    data = fetch_data(ticker, start_date, end_date)
    
    if data.empty:
        return None
    
    try:
        output = run_backtest(SmaCross, data, cash, commission)
        return output
    except Exception as e:
        st.error(f"An error occurred during backtesting: {str(e)}")
        return None
    
    