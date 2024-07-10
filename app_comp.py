import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go

from data_handler import fetch_data
from utils import display_metrics, plot_strat_perf, run_backtest
from strategies.sma_cross import SmaCross, sma_cross_viz, run_sma_cross
from strategies.rsi_cross import RsiCross, rsi_cross_viz, run_rsi_cross
from strategies.bollinger_bands import BollingerBandsStrategy, bollinger_bands_viz, run_bollinger_bands
from strategies.macd import MACDStrategy, macd_viz, run_macd_strategy
from strategies.vwap import VWAPStrategy, vwap_viz, run_vwap_strategy
from strategies.stochastic import StochStrategy, stoch_viz, run_stoch_strategy
from strategies.mean_reversion import MeanReversion, mean_reversion_viz, run_mean_reversion
from strategies.momentum import MomentumStrategy, momentum_viz, run_momentum
from strategies.adx import ADXStrategy, adx_viz, run_adx
from strategies.cci import CCIStrategy, cci_viz, run_cci
from strategies.dpo import DPOStrategy, dpo_viz, run_dpo
from strategies.obv import OBVStrategy, obv_viz, run_obv_strategy
from strategies.atr import ATRStrategy, atr_viz, run_atr_strategy
from strategies.standard_deviation import StdDevStrategy, std_dev_viz, run_std_dev_strategy

st.set_page_config(layout="wide", page_title="Little John - Strategy Analyzer and Comparator")

def run_strategy(strategy, ticker, start_date, end_date, cash, commission, **params):
    strategy_functions = {
        'SMA Cross': run_sma_cross,
        'RSI Cross': run_rsi_cross,
        'Bollinger Bands': run_bollinger_bands,
        'MACD': run_macd_strategy,
        'VWAP': run_vwap_strategy,
        'Stochastic': run_stoch_strategy,
        'Mean Reversion': run_mean_reversion,
        'Momentum': run_momentum,
        'ADX': run_adx,
        'CCI': run_cci,
        'DPO': run_dpo,
        'OBV': run_obv_strategy,
        'ATR': run_atr_strategy,
        'Standard Deviation': run_std_dev_strategy
    }
    
    return strategy_functions[strategy](ticker, start_date, end_date, cash, commission, **params)

def strategy_parameters(strategy):
    params = {}
    if strategy == 'SMA Cross':
        sma_short = st.slider('Short SMA', min_value=5, max_value=50, value=10, key=f'{strategy}_short')
        sma_long = st.slider('Long SMA', min_value=10, max_value=100, value=20, key=f'{strategy}_long')
        return {'sma_short': sma_short, 'sma_long': sma_long}
    
    elif strategy == 'RSI Cross':
        st.subheader('RSI Cross Parameters')
        rsi_period = st.slider('RSI Period', min_value=2, max_value=30, value=14)
        rsi_sma_short = st.slider('RSI Short SMA', min_value=5, max_value=50, value=10, key='rsi_short')
        rsi_sma_long = st.slider('RSI Long SMA', min_value=10, max_value=100, value=20, key='rsi_long')

    elif strategy == 'Bollinger Bands':
        st.subheader('Bollinger Bands Parameters')
        bb_period = st.slider('MA Period', min_value=5, max_value=50, value=20, key='bb_period')
        bb_std_dev = st.slider('Std Dev Multiplier', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='bb_std_dev')

    elif strategy == 'MACD':
        st.subheader('MACD Parameters')
        macd_fast = st.slider('Fast Period', min_value=5, max_value=50, value=12, key='macd_fast')
        macd_slow = st.slider('Slow Period', min_value=10, max_value=100, value=26, key='macd_slow')
        macd_signal = st.slider('Signal Period', min_value=5, max_value=50, value=9, key='macd_signal')

    elif strategy == 'VWAP':
        st.subheader('VWAP Parameters')
        vwap_periods = st.slider('VWAP Periods', min_value=5, max_value=50, value=20, key='vwap_periods')

    elif strategy == 'Stochastic':
        st.subheader('Stochastic Parameters')
        stoch_k = st.slider('K Period', min_value=5, max_value=50, value=14, key='stoch_k')
        stoch_d = st.slider('D Period', min_value=1, max_value=10, value=3, key='stoch_d')
        stoch_overbought = st.slider('Overbought Level', min_value=50, max_value=95, value=80, key='stoch_overbought')
        stoch_oversold = st.slider('Oversold Level', min_value=5, max_value=50, value=20, key='stoch_oversold')

    elif strategy == 'Mean Reversion':
        st.subheader('Mean Reversion Parameters')
        mr_period = st.slider('Lookback Period', min_value=5, max_value=50, value=20, key='mr_period')
        mr_entry_std = st.slider('Entry Std Dev', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='mr_entry_std')
        mr_exit_std = st.slider('Exit Std Dev', min_value=0.1, max_value=2.0, value=0.5, step=0.1, key='mr_exit_std')

    elif strategy == 'Momentum':
        st.subheader('Momentum Parameters')
        mom_period = st.slider('ROC Period', min_value=5, max_value=50, value=14, key='mom_period')
        mom_threshold = st.slider('ROC Threshold', min_value=0.0, max_value=5.0, value=2.0, step=0.1, key='mom_threshold')

    elif strategy == 'ADX':
        st.subheader('ADX Parameters')
        adx_period = st.slider('ADX Period', min_value=5, max_value=50, value=14, key='adx_period')
        adx_threshold = st.slider('ADX Threshold', min_value=10, max_value=50, value=25, key='adx_threshold')

    elif strategy == 'CCI':
        st.subheader('CCI Parameters')
        cci_period = st.slider('CCI Period', min_value=5, max_value=50, value=20, key='cci_period')
        cci_overbought = st.slider('Overbought Level', min_value=50, max_value=200, value=100, key='cci_overbought')
        cci_oversold = st.slider('Oversold Level', min_value=-200, max_value=-50, value=-100, key='cci_oversold')

    elif strategy == 'DPO':
        st.subheader('DPO Parameters')
        dpo_period = st.slider('DPO Period', min_value=5, max_value=50, value=20, key='dpo_period')
        dpo_threshold = st.slider('DPO Threshold', min_value=0.0, max_value=5.0, value=0.5, step=0.1, key='dpo_threshold')

    elif strategy == 'OBV':
        st.subheader('OBV Parameters')
        obv_periods = st.slider('OBV SMA Periods', min_value=5, max_value=50, value=20, key='obv_periods')

    elif strategy == 'ATR':
        st.subheader('ATR Parameters')
        atr_period = st.slider('ATR Period', min_value=5, max_value=50, value=14, key='atr_period')
        atr_multiplier = st.slider('ATR Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='atr_multiplier')

    elif strategy == 'Standard Deviation':
        st.subheader('Standard Deviation Parameters')
        std_period = st.slider('Period', min_value=5, max_value=50, value=20, key='std_period')
        std_multiplier = st.slider('Std Dev Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='std_multiplier')
    
    else:
        st.warning(f"No specific parameters for {strategy}. Using default values.")
    
    # Common parameters for all strategies
    params['stop_loss_pct'] = st.slider('Stop Loss %', min_value=0.0, max_value=10.0, value=2.0, step=0.1, key=f'{strategy}_stop_loss')
    params['take_profit_pct'] = st.slider('Take Profit %', min_value=0.0, max_value=10.0, value=5.0, step=0.1, key=f'{strategy}_take_profit')
    params['enable_stop_loss'] = st.checkbox('Enable Stop Loss', value=True, key=f'{strategy}_enable_stop_loss')
    params['enable_take_profit'] = st.checkbox('Enable Take Profit', value=True, key=f'{strategy}_enable_take_profit')
    params['enable_shorting'] = st.checkbox('Enable Shorting', value=True, key=f'{strategy}_enable_shorting')
    
    return params


def display_metrics_comparison(output1, output2):
    metrics1 = display_metrics(output1)
    metrics2 = display_metrics(output2)
    
    df = pd.DataFrame({
        'Strategy 1': metrics1,
        'Strategy 2': metrics2
    })
    
    st.dataframe(df)
    
    # Highlight key metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return 1 vs 2", 
                f"{metrics1['Return [%]']:.2f}% vs {metrics2['Return [%]']:.2f}%",
                f"{metrics1['Return [%]'] - metrics2['Return [%]']:.2f}%")
    col2.metric("Sharpe Ratio 1 vs 2", 
                f"{metrics1['Sharpe Ratio']:.2f} vs {metrics2['Sharpe Ratio']:.2f}",
                f"{metrics1['Sharpe Ratio'] - metrics2['Sharpe Ratio']:.2f}")
    col3.metric("Max Drawdown 1 vs 2", 
                f"{metrics1['Max. Drawdown [%]']:.2f}% vs {metrics2['Max. Drawdown [%]']:.2f}%",
                f"{metrics2['Max. Drawdown [%]'] - metrics1['Max. Drawdown [%]']:.2f}%")
    col4.metric("Win Rate 1 vs 2", 
                f"{metrics1['Win Rate [%]']:.2f}% vs {metrics2['Win Rate [%]']:.2f}%",
                f"{metrics1['Win Rate [%]'] - metrics2['Win Rate [%]']:.2f}%")

def plot_equity_curves(output1, output2, strategy1, strategy2):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=output1['_equity_curve'].index, y=output1['_equity_curve']['Equity'],
                             mode='lines', name=f'{strategy1} Equity'))
    fig.add_trace(go.Scatter(x=output2['_equity_curve'].index, y=output2['_equity_curve']['Equity'],
                             mode='lines', name=f'{strategy2} Equity'))
    fig.update_layout(title='Equity Curves Comparison',
                      xaxis_title='Date',
                      yaxis_title='Equity')
    st.plotly_chart(fig)

def main():
    st.title('Little John - Strategy Analyzer and Comparator')
    
    page = st.sidebar.selectbox("Choose a page", ["Individual Strategy", "Strategy Comparison"])
    
    if page == "Individual Strategy":
        individual_strategy_page()
    elif page == "Strategy Comparison":
        strategy_comparison_page()

def individual_strategy_page():
    st.header("Individual Strategy Analysis")
    col1, col2 = st.columns([1, 2])

    with col1:
        ticker = st.text_input('Enter stock ticker', value='AAPL')

        strategy_groups = {
            'Trend': ['SMA Cross', 'ADX', 'DPO', 'CCI'],
            'Momentum': ['RSI Cross', 'MACD', 'Stochastic', 'Momentum'],
            'Volume': ['VWAP', 'OBV'],
            'Volatility': ['Bollinger Bands', 'ATR', 'Standard Deviation'],
            'Other': ['Mean Reversion']
        }
        with st.expander("Strategy", expanded=True):
            group = st.selectbox('Select Strategy Type', list(strategy_groups.keys()))
            strategy = st.selectbox('Strategy', strategy_groups[group])

        sixty_days_ago = datetime.now() - timedelta(days=59)
        ten_days_ago = datetime.now() - timedelta(days=10)

        with st.expander("Common Parameters", expanded=True):
            start_date = st.date_input('Start Date', 
                                    value=ten_days_ago,
                                    min_value=sixty_days_ago,
                                    max_value=datetime.now())
            end_date = st.date_input('End Date', value=datetime.now())
            cash = st.number_input('Initial Cash', min_value=1000, max_value=1000000, value=10000)
            commission = st.slider('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)

        # Strategy-specific parameters
        if strategy == 'SMA Cross':
            st.subheader('SMA Cross Parameters')
            sma_short = st.slider('Short SMA', min_value=5, max_value=50, value=10, key='sma_short')
            sma_long = st.slider('Long SMA', min_value=10, max_value=100, value=20, key='sma_long')
        
        elif strategy == 'RSI Cross':
            st.subheader('RSI Cross Parameters')
            rsi_period = st.slider('RSI Period', min_value=2, max_value=30, value=14)
            rsi_sma_short = st.slider('RSI Short SMA', min_value=5, max_value=50, value=10, key='rsi_short')
            rsi_sma_long = st.slider('RSI Long SMA', min_value=10, max_value=100, value=20, key='rsi_long')

        elif strategy == 'Bollinger Bands':
            st.subheader('Bollinger Bands Parameters')
            bb_period = st.slider('MA Period', min_value=5, max_value=50, value=20, key='bb_period')
            bb_std_dev = st.slider('Std Dev Multiplier', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='bb_std_dev')

        elif strategy == 'MACD':
            st.subheader('MACD Parameters')
            macd_fast = st.slider('Fast Period', min_value=5, max_value=50, value=12, key='macd_fast')
            macd_slow = st.slider('Slow Period', min_value=10, max_value=100, value=26, key='macd_slow')
            macd_signal = st.slider('Signal Period', min_value=5, max_value=50, value=9, key='macd_signal')

        elif strategy == 'VWAP':
            st.subheader('VWAP Parameters')
            vwap_periods = st.slider('VWAP Periods', min_value=5, max_value=50, value=20, key='vwap_periods')

        elif strategy == 'Stochastic':
            st.subheader('Stochastic Parameters')
            stoch_k = st.slider('K Period', min_value=5, max_value=50, value=14, key='stoch_k')
            stoch_d = st.slider('D Period', min_value=1, max_value=10, value=3, key='stoch_d')
            stoch_overbought = st.slider('Overbought Level', min_value=50, max_value=95, value=80, key='stoch_overbought')
            stoch_oversold = st.slider('Oversold Level', min_value=5, max_value=50, value=20, key='stoch_oversold')

        elif strategy == 'Mean Reversion':
            st.subheader('Mean Reversion Parameters')
            mr_period = st.slider('Lookback Period', min_value=5, max_value=50, value=20, key='mr_period')
            mr_entry_std = st.slider('Entry Std Dev', min_value=0.5, max_value=3.0, value=2.0, step=0.1, key='mr_entry_std')
            mr_exit_std = st.slider('Exit Std Dev', min_value=0.1, max_value=2.0, value=0.5, step=0.1, key='mr_exit_std')

        elif strategy == 'Momentum':
            st.subheader('Momentum Parameters')
            mom_period = st.slider('ROC Period', min_value=5, max_value=50, value=14, key='mom_period')
            mom_threshold = st.slider('ROC Threshold', min_value=0.0, max_value=5.0, value=2.0, step=0.1, key='mom_threshold')

        elif strategy == 'ADX':
            st.subheader('ADX Parameters')
            adx_period = st.slider('ADX Period', min_value=5, max_value=50, value=14, key='adx_period')
            adx_threshold = st.slider('ADX Threshold', min_value=10, max_value=50, value=25, key='adx_threshold')



        elif strategy == 'CCI':
            st.subheader('CCI Parameters')
            cci_period = st.slider('CCI Period', min_value=5, max_value=50, value=20, key='cci_period')
            cci_overbought = st.slider('Overbought Level', min_value=50, max_value=200, value=100, key='cci_overbought')
            cci_oversold = st.slider('Oversold Level', min_value=-200, max_value=-50, value=-100, key='cci_oversold')

        elif strategy == 'DPO':
            st.subheader('DPO Parameters')
            dpo_period = st.slider('DPO Period', min_value=5, max_value=50, value=20, key='dpo_period')
            dpo_threshold = st.slider('DPO Threshold', min_value=0.0, max_value=5.0, value=0.5, step=0.1, key='dpo_threshold')

        elif strategy == 'OBV':
            st.subheader('OBV Parameters')
            obv_periods = st.slider('OBV SMA Periods', min_value=5, max_value=50, value=20, key='obv_periods')

        elif strategy == 'ATR':
            st.subheader('ATR Parameters')
            atr_period = st.slider('ATR Period', min_value=5, max_value=50, value=14, key='atr_period')
            atr_multiplier = st.slider('ATR Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='atr_multiplier')

        elif strategy == 'Standard Deviation':
            st.subheader('Standard Deviation Parameters')
            std_period = st.slider('Period', min_value=5, max_value=50, value=20, key='std_period')
            std_multiplier = st.slider('Std Dev Multiplier', min_value=1.0, max_value=5.0, value=2.0, step=0.1, key='std_multiplier')
        
        
        
        enable_shorting = st.checkbox('Enable Shorting', value=True)
            
        with st.expander("Stop Loss / Take Profit", expanded=True):
            stop_loss_pct = st.slider('Stop Loss %', min_value=0.0, max_value=10.0, value=2.0, step=0.1)
            take_profit_pct = st.slider('Take Profit %', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
            enable_stop_loss = st.checkbox('Enable Stop Loss', value=True)
            enable_take_profit = st.checkbox('Enable Take Profit', value=True)

    with col2:
        #with st.spinner('Running backtest...'):
            data = fetch_data(ticker, start_date, end_date)
            
            if data.empty:
                st.warning("No data available for the selected date range.")
            else:
                if strategy == 'SMA Cross':
                    st.subheader('SMA Cross Visualization')
                    sma_cross_viz(data, sma_short, sma_long)
                    output = run_sma_cross(ticker, start_date, end_date, cash, commission, sma_short, sma_long, 
                                           stop_loss_pct, take_profit_pct, enable_shorting, 
                                           enable_stop_loss, enable_take_profit)

                elif strategy == 'RSI Cross':
                    st.subheader('RSI Cross Visualization')
                    rsi_cross_viz(data, rsi_sma_short, rsi_sma_long, rsi_period)
                    output = run_rsi_cross(ticker, start_date, end_date, cash, commission, rsi_sma_short, rsi_sma_long, 
                                           rsi_period, stop_loss_pct, take_profit_pct, enable_shorting, 
                                           enable_stop_loss, enable_take_profit)

                elif strategy == 'Bollinger Bands':
                    st.subheader('Bollinger Bands Visualization')
                    bollinger_bands_viz(data, bb_period, bb_std_dev)
                    output = run_bollinger_bands(ticker, start_date, end_date, cash, commission, bb_period, bb_std_dev,
                                                 stop_loss_pct, take_profit_pct, enable_shorting, 
                                                 enable_stop_loss, enable_take_profit)

                elif strategy == 'MACD':
                    st.subheader('MACD Visualization')
                    macd_viz(data, macd_fast, macd_slow, macd_signal)
                    output = run_macd_strategy(ticker, start_date, end_date, cash, commission, macd_fast, macd_slow, macd_signal,
                                               stop_loss_pct, take_profit_pct, enable_shorting, 
                                               enable_stop_loss, enable_take_profit)

                elif strategy == 'VWAP':
                    st.subheader('VWAP Visualization')
                    vwap_viz(data, vwap_periods)
                    output = run_vwap_strategy(ticker, start_date, end_date, cash, commission, vwap_periods,
                                               stop_loss_pct, take_profit_pct, enable_shorting, 
                                               enable_stop_loss, enable_take_profit)

                elif strategy == 'Stochastic':
                    st.subheader('Stochastic Visualization')
                    stoch_viz(data, stoch_k, stoch_d, stoch_overbought, stoch_oversold)
                    output = run_stoch_strategy(ticker, start_date, end_date, cash, commission, stoch_k, stoch_d, 
                                                stoch_overbought, stoch_oversold, stop_loss_pct, take_profit_pct, 
                                                enable_shorting, enable_stop_loss, enable_take_profit)
                    
                elif strategy == 'Mean Reversion':
                    st.subheader('Mean Reversion Visualization')
                    mean_reversion_viz(data, mr_period, mr_entry_std, mr_exit_std)
                    output = run_mean_reversion(ticker, start_date, end_date, cash, commission, mr_period, mr_entry_std, 
                                                mr_exit_std, stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, 
                                                enable_take_profit)                    

                elif strategy == 'Momentum':
                    st.subheader('Momentum Visualization')
                    data = fetch_data(ticker, start_date, end_date)
                    if not data.empty:
                        momentum_viz(data, mom_period, mom_threshold)
                        output = run_momentum(ticker, start_date, end_date, cash, commission, mom_period, mom_threshold, 
                                            stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit)
                    else:
                        st.warning("No data available for the selected date range.")

                elif strategy == 'ADX':                    
                    st.subheader('ADX Visualization')
                    data = fetch_data(ticker, start_date, end_date)
                    if not data.empty:
                        adx_viz(data, adx_period, adx_threshold)
                        output = run_adx(ticker, start_date, end_date, cash, commission, adx_period, adx_threshold, 
                                        stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit)
                    else:
                        st.warning("No data available for the selected date range.")
                    
                elif strategy == 'CCI': 
                    st.subheader('CCI Visualization')
                    cci_viz(data, cci_period)
                    output = run_cci(ticker, start_date, end_date, cash, commission, cci_period, cci_overbought, cci_oversold, 
                                        stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit)

                elif strategy == 'DPO':
                    st.subheader('DPO Visualization')
                    data = fetch_data(ticker, start_date, end_date)
                    if not data.empty:
                        output = run_dpo(ticker, start_date, end_date, cash, commission, dpo_period, dpo_threshold,
                                        stop_loss_pct, take_profit_pct, enable_shorting, enable_stop_loss, enable_take_profit)
                    else:
                        st.warning("No data available for the selected date range.")
                    
                elif strategy == 'OBV':
                    st.subheader('OBV Visualization')
                    obv_viz(data, obv_periods)
                    output = run_obv_strategy(ticker, start_date, end_date, cash, commission, obv_periods,
                                            stop_loss_pct, take_profit_pct, enable_shorting, 
                                            enable_stop_loss, enable_take_profit)
                
                elif strategy == 'ATR':
                    st.subheader('ATR Visualization')
                    atr_viz(data, atr_period, atr_multiplier)
                    output = run_atr_strategy(ticker, start_date, end_date, cash, commission, atr_period, atr_multiplier,
                                            stop_loss_pct, take_profit_pct, enable_shorting, 
                                            enable_stop_loss, enable_take_profit)

                elif strategy == 'Standard Deviation':
                    st.subheader('Standard Deviation Visualization')
                    std_dev_viz(data, std_period, std_multiplier)
                    output = run_std_dev_strategy(ticker, start_date, end_date, cash, commission, std_period, std_multiplier,
                                                stop_loss_pct, take_profit_pct, enable_shorting, 
                                                enable_stop_loss, enable_take_profit)
                
            if output is not None:
                st.subheader('Strategy Performance')
                plot_strat_perf(output, f"{strategy} Strategy Performance - {ticker}")
                
                st.subheader('Key Performance Metrics')
                metrics = display_metrics(output)
                metrics_df = pd.DataFrame([metrics]).T
                metrics_df.columns = ['Value']
                st.dataframe(metrics_df)

                # Display key metrics in a more prominent way
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Return", f"{metrics['Return [%]']:.2f}%")
                col3.metric("Max Drawdown", f"{metrics['Max. Drawdown [%]']:.2f}%")
                col2.metric("Win Rate", f"{metrics['Win Rate [%]']:.2f}%")
                col1.metric("Best Trade", f"{metrics['Best Trade [%]']:.2f}%")
                col1.metric("Worst Trade", f"{metrics['Worst Trade [%]']:.2f}%")


                st.subheader('Trade Log')
                if '_trades' in output:
                    trades_df = output['_trades']
                    
                    # List of columns we want to display if they're available
                    columns_to_display = ['Size', 'PnL', 'ReturnPct', 'EntryBar', 'ExitBar', 'EntryPrice', 'ExitPrice', 'EntryTime', 'ExitTime', 'Duration']
                    
                    # Filter the DataFrame to only include available columns
                    trades_df_display = trades_df[[col for col in columns_to_display if col in trades_df.columns]]
                    
                    # Format specific columns if they exist
                    if 'ReturnPct' in trades_df_display.columns:
                        trades_df_display['ReturnPct'] = trades_df_display['ReturnPct'].apply(lambda x: f"{x:.2f}%")
                    if 'Duration' in trades_df_display.columns:
                        trades_df_display['Duration'] = trades_df_display['Duration'].apply(lambda x: str(x))
                    if 'PnL' in trades_df_display.columns:
                        trades_df_display['PnL'] = trades_df_display['PnL'].apply(lambda x: f"{x:.2f}")
                    
                    # Display the trade log
                    st.dataframe(trades_df_display)
                    
                    # Display summary statistics
                    st.subheader('Trade Summary')
                    summary = pd.DataFrame({
                        'Total Trades': len(trades_df),
                        'Profitable Trades': (trades_df['PnL'] > 0).sum() if 'PnL' in trades_df.columns else 'N/A',
                        'Loss-Making Trades': (trades_df['PnL'] < 0).sum() if 'PnL' in trades_df.columns else 'N/A',
                        'Total PnL': trades_df['PnL'].sum() if 'PnL' in trades_df.columns else 'N/A',
                        'Average PnL per Trade': trades_df['PnL'].mean() if 'PnL' in trades_df.columns else 'N/A',
                    }, index=['Value'])
                    st.dataframe(summary.T)
                else:
                    st.write("No trade data available.")
            else:
                st.warning("Backtest did not complete successfully. Please check your parameters.")

def strategy_comparison_page():
    st.header("Strategy Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy 1")
        ticker1 = st.text_input('Enter stock ticker', value='AAPL', key='ticker1')
        strategy1 = st.selectbox('Select Strategy', ['SMA Cross', 'RSI Cross', 'Bollinger Bands', 'MACD', 'VWAP', 'Stochastic', 'Mean Reversion', 'Momentum', 'ADX', 'CCI', 'DPO', 'OBV', 'ATR', 'Standard Deviation'], key='strategy1')
        params1 = strategy_parameters(strategy1)
    
    with col2:
        st.subheader("Strategy 2")
        ticker2 = st.text_input('Enter stock ticker', value='AAPL', key='ticker2')
        strategy2 = st.selectbox('Select Strategy', ['SMA Cross', 'RSI Cross', 'Bollinger Bands', 'MACD', 'VWAP', 'Stochastic', 'Mean Reversion', 'Momentum', 'ADX', 'CCI', 'DPO', 'OBV', 'ATR', 'Standard Deviation'], key='strategy2')
        params2 = strategy_parameters(strategy2)
    
    # Common parameters
    start_date = st.date_input('Start Date', value=datetime.now() - timedelta(days=365))
    end_date = st.date_input('End Date', value=datetime.now())
    cash = st.number_input('Initial Cash', min_value=1000, max_value=1000000, value=10000)
    commission = st.slider('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    
    if st.button('Run Comparison'):
        with st.spinner('Running backtests...'):
            params1 = strategy_parameters(strategy1)
            params2 = strategy_parameters(strategy2)
            
            output1 = run_strategy(strategy1, ticker1, start_date, end_date, cash, commission, **params1)
            output2 = run_strategy(strategy2, ticker2, start_date, end_date, cash, commission, **params2)
        
        
        if output1 is not None and output2 is not None:
            st.subheader('Performance Comparison')
            display_metrics_comparison(output1, output2)
            
            st.subheader('Equity Curves Comparison')
            plot_equity_curves(output1, output2, strategy1, strategy2)
            
            st.subheader('Trade Logs')
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"{strategy1} Trade Log")
                if '_trades' in output1:
                    st.dataframe(output1['_trades'])
                else:
                    st.write("No trade data available.")
            with col2:
                st.write(f"{strategy2} Trade Log")
                if '_trades' in output2:
                    st.dataframe(output2['_trades'])
                else:
                    st.write("No trade data available.")
        else:
            st.warning("One or both backtests did not complete successfully. Please check your parameters.")

if __name__ == "__main__":
    main()