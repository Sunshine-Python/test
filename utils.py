import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
import matplotlib.dates as mdates
import streamlit as st


def display_metrics(output):
    metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
               'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
               'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
               'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
               'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
               'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]', 'Number of Trades']
    
    result = {k: output[k] for k in metrics if k in output}
    
    # Add Number of Trades explicitly if it's not already in the output
    if 'Number of Trades' not in result and 'Trades' in result:
        result['Number of Trades'] = result['Trades']
    
    return result


def plot_strat_perf(output, title):
    if '_equity_curve' not in output:
        st.error("Equity curve data not available. The backtest may not have completed successfully.")
        return

    equity_curve = output['_equity_curve']
    
    if equity_curve.empty:
        st.warning("Equity curve is empty. No trades may have been executed.")
        return

    # Filter for trading days at market close (4:00 PM)
    trading_day_equity = equity_curve[
        (equity_curve.index.dayofweek < 5) &  # Monday = 0, Friday = 4
        (equity_curve.index.hour == 15) &     # 3:00 PM (15:00)
        (equity_curve.index.minute == 55)     # Last data point before 4:00 PM
    ]
    
    if trading_day_equity.empty:
        st.warning("No data points match the filtering criteria. Displaying full equity curve.")
        trading_day_equity = equity_curve

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(trading_day_equity.index, trading_day_equity['Equity'], label='Equity')
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.legend()
    ax.grid(True)
    
    fig.autofmt_xdate()
    
    # Format the x-axis to display dates in YYYY-MM-DD format
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    plt.tight_layout()
    st.pyplot(fig)



def run_backtest(strategy, data, cash=10000, commission=0.1):
    commission_decimal = commission / 100  # Convert percentage to decimal
    bt = Backtest(data, strategy, cash=cash, commission=commission_decimal)
    output = bt.run()
    return output





def display_performance_metrics(output):
    """
    Displays key performance metrics.
    """
    st.subheader('Performance Metrics')
    
    key_metrics = ['Start', 'End', 'Duration', 'Exposure Time [%]', 'Equity Final [$]', 'Equity Peak [$]', 
                   'Return [%]', 'Buy & Hold Return [%]', 'Return (Ann.) [%]', 'Volatility (Ann.) [%]', 
                   'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max. Drawdown [%]', 'Avg. Drawdown [%]', 
                   'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Trades', 'Win Rate [%]', 
                   'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]', 'Max. Trade Duration', 
                   'Avg. Trade Duration', 'Profit Factor', 'Expectancy [%]']
    
    metrics = output.drop(['_strategy', '_equity_curve', '_trades'])
    selected_metrics = {k: metrics[k] for k in key_metrics if k in metrics}
    df_metrics = pd.DataFrame(selected_metrics, index=['Value']).T
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{df_metrics.loc['Return [%]', 'Value']:.2f}%")
    col2.metric("Sharpe Ratio", f"{df_metrics.loc['Sharpe Ratio', 'Value']:.2f}")
    col3.metric("Max Drawdown", f"{df_metrics.loc['Max. Drawdown [%]', 'Value']:.2f}%")
    
    strategy_return = df_metrics.loc['Return [%]', 'Value']
    bh_return = df_metrics.loc['Buy & Hold Return [%]', 'Value']
    outperformance = strategy_return - bh_return
    st.metric("Strategy vs. Buy & Hold", f"{outperformance:.2f}%", 
              delta=f"{outperformance:.2f}%", delta_color="normal")
    
    st.metric("Win Rate", f"{df_metrics.loc['Win Rate [%]', 'Value']:.2f}%")
    
    with st.expander("View All Metrics"):
        st.dataframe(df_metrics, use_container_width=True)

def plot_return_comparison(strategy_return, bh_return):
    """
    Plots a bar chart comparing strategy return to buy & hold return.
    """
    fig_return_comparison = go.Figure(data=[
        go.Bar(name='Strategy', x=['Return'], y=[strategy_return]),
        go.Bar(name='Buy & Hold', x=['Return'], y=[bh_return])
    ])
    fig_return_comparison.update_layout(title='Strategy vs. Buy & Hold Return Comparison')
    st.plotly_chart(fig_return_comparison, use_container_width=True)
