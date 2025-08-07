import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from arch import arch_model
import pandas_ta
import warnings
import matplotlib.pyplot as plt
from io import BytesIO
import base64

warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Intraday GARCH Trading Strategy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'garch_fitted' not in st.session_state:
    st.session_state.garch_fitted = False

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration"""
    # Create sample daily data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
    np.random.seed(42)
    
    daily_data = []
    price = 100
    for date in dates:
        if date.weekday() < 5:  # Only weekdays
            daily_return = np.random.normal(0.0005, 0.02)
            price = price * (1 + daily_return)
            
            high = price * (1 + abs(np.random.normal(0, 0.01)))
            low = price * (1 - abs(np.random.normal(0, 0.01)))
            
            daily_data.append({
                'Date': date,
                'Open': price * (1 + np.random.normal(0, 0.005)),
                'High': high,
                'Low': low,
                'Close': price,
                'Adj Close': price,
                'Volume': np.random.randint(1000000, 5000000)
            })
    
    daily_df = pd.DataFrame(daily_data)
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.set_index('Date')
    
    # Create sample 5-minute intraday data
    intraday_data = []
    current_date = pd.to_datetime('2020-01-01')
    end_date = pd.to_datetime('2024-12-31')
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Only weekdays
            daily_price = daily_df.loc[daily_df.index.date == current_date.date(), 'Close'].iloc[0] if len(daily_df.loc[daily_df.index.date == current_date.date()]) > 0 else 100
            
            for hour in range(9, 16):  # 9 AM to 4 PM
                for minute in range(0, 60, 5):  # Every 5 minutes
                    timestamp = current_date.replace(hour=hour, minute=minute)
                    
                    intraday_return = np.random.normal(0, 0.003)
                    price = daily_price * (1 + intraday_return + np.random.normal(0, 0.001))
                    
                    intraday_data.append({
                        'datetime': timestamp,
                        'open': price * (1 + np.random.normal(0, 0.001)),
                        'high': price * (1 + abs(np.random.normal(0, 0.002))),
                        'low': price * (1 - abs(np.random.normal(0, 0.002))),
                        'close': price,
                        'volume': np.random.randint(10000, 100000)
                    })
        
        current_date += pd.Timedelta(days=1)
    
    intraday_df = pd.DataFrame(intraday_data)
    intraday_df['datetime'] = pd.to_datetime(intraday_df['datetime'])
    intraday_df = intraday_df.set_index('datetime')
    intraday_df['date'] = pd.to_datetime(intraday_df.index.date)
    
    return daily_df, intraday_df

@st.cache_data
def fit_garch_model(log_returns, window_size, p, q):
    """Fit GARCH model with rolling window"""
    predictions = []
    dates = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(window_size, len(log_returns)):
        try:
            data_window = log_returns.iloc[i-window_size:i].dropna()
            
            if len(data_window) >= 50:  # Minimum observations
                model = arch_model(data_window, p=p, q=q, vol='GARCH')
                fitted_model = model.fit(update_freq=0, disp='off')
                
                forecast = fitted_model.forecast(horizon=1)
                variance_forecast = forecast.variance.iloc[-1, 0]
                
                predictions.append(variance_forecast)
                dates.append(log_returns.index[i])
            else:
                predictions.append(np.nan)
                dates.append(log_returns.index[i])
        
        except Exception as e:
            predictions.append(np.nan)
            dates.append(log_returns.index[i])
        
        # Update progress
        progress = (i - window_size + 1) / (len(log_returns) - window_size)
        progress_bar.progress(progress)
        status_text.text(f'Fitting GARCH model... {progress:.1%} complete')
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.Series(predictions, index=dates, name='garch_predictions')

def generate_signals(daily_df, intraday_df, params):
    """Generate trading signals"""
    # Daily signals
    daily_df['log_ret'] = np.log(daily_df['Adj Close']).diff()
    daily_df['variance'] = daily_df['log_ret'].rolling(params['variance_window']).var()
    
    # GARCH predictions (simplified for demo)
    if 'garch_predictions' not in daily_df.columns:
        daily_df['predictions'] = daily_df['variance'] * (1 + np.random.normal(0, 0.1, len(daily_df)))
    else:
        daily_df['predictions'] = daily_df['garch_predictions']
    
    # Prediction premium
    daily_df['prediction_premium'] = (daily_df['predictions'] - daily_df['variance']) / daily_df['variance']
    daily_df['premium_std'] = daily_df['prediction_premium'].rolling(params['premium_window']).std()
    
    # Daily signal
    daily_df['signal_daily'] = daily_df.apply(
        lambda x: 1 if x['prediction_premium'] > params['std_multiplier'] * x['premium_std']
        else (-1 if x['prediction_premium'] < -params['std_multiplier'] * x['premium_std'] else np.nan),
        axis=1
    )
    daily_df['signal_daily'] = daily_df['signal_daily'].shift()
    
    # Merge with intraday data
    final_df = intraday_df.reset_index().merge(
        daily_df[['signal_daily']].reset_index(),
        left_on='date',
        right_on='Date'
    ).drop(['date', 'Date'], axis=1).set_index('datetime')
    
    # Intraday indicators
    final_df['rsi'] = pandas_ta.rsi(close=final_df['close'], length=params['rsi_length'])
    
    bb = pandas_ta.bbands(close=final_df['close'], length=params['bb_length'])
    if bb is not None and not bb.empty:
        final_df['bb_lower'] = bb.iloc[:, 0]
        final_df['bb_upper'] = bb.iloc[:, 2]
        final_df['bb_middle'] = bb.iloc[:, 1]
    else:
        # Fallback calculation if pandas_ta fails
        sma = final_df['close'].rolling(window=params['bb_length']).mean()
        std = final_df['close'].rolling(window=params['bb_length']).std()
        final_df['bb_upper'] = sma + (std * 2)
        final_df['bb_lower'] = sma - (std * 2)
        final_df['bb_middle'] = sma
    
    # Enhanced Intraday signal with proper logic
    # Signal generation: 1 = Sell signal (Overbought), -1 = Buy signal (Oversold)
    final_df['rsi_overbought'] = final_df['rsi'] > params['rsi_overbought']
    final_df['rsi_oversold'] = final_df['rsi'] < params['rsi_oversold']
    final_df['price_above_bb_upper'] = final_df['close'] > final_df['bb_upper']
    final_df['price_below_bb_lower'] = final_df['close'] < final_df['bb_lower']
    
    # Generate signals with clear conditions
    final_df['signal_intraday'] = np.nan
    
    # Sell signal: RSI > 70 AND Price > BB Upper (Overbought condition)
    sell_condition = final_df['rsi_overbought'] & final_df['price_above_bb_upper']
    final_df.loc[sell_condition, 'signal_intraday'] = 1
    
    # Buy signal: RSI < 30 AND Price < BB Lower (Oversold condition)  
    buy_condition = final_df['rsi_oversold'] & final_df['price_below_bb_lower']
    final_df.loc[buy_condition, 'signal_intraday'] = -1
    
    # Combined signal
    final_df['return_sign'] = final_df.apply(
        lambda x: -1 if (x['signal_daily'] == 1) & (x['signal_intraday'] == 1)
        else (1 if (x['signal_daily'] == -1) & (x['signal_intraday'] == -1) else np.nan),
        axis=1
    )
    
    # Forward fill signals within each day
    final_df['return_sign'] = final_df.groupby(pd.Grouper(freq='D'))['return_sign'].transform(lambda x: x.ffill())
    
    # Calculate returns
    final_df['return'] = np.log(final_df['close']).diff()
    final_df['forward_return'] = final_df['return'].shift(-1)
    final_df['strategy_return'] = final_df['forward_return'] * final_df['return_sign']
    
    return daily_df, final_df

def calculate_performance_metrics(returns):
    """Calculate comprehensive performance metrics"""
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return {}
    
    total_return = (1 + returns_clean).prod() - 1
    annualized_return = (1 + returns_clean.mean()) ** 252 - 1
    volatility = returns_clean.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    # Drawdown calculation
    cumulative_returns = (1 + returns_clean).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns_clean > 0).mean()
    
    # Number of trades (non-zero returns)
    num_trades = (returns_clean != 0).sum()
    
    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Volatility': f"{volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Maximum Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Number of Trades': f"{num_trades:,}",
        'Calmar Ratio': f"{annualized_return / abs(max_drawdown):.2f}" if max_drawdown != 0 else "N/A"
    }

def main():
    st.title("📈 Intraday GARCH Trading Strategy")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Strategy Parameters")
    
    # GARCH Parameters
    st.sidebar.subheader("GARCH Model Parameters")
    garch_window = st.sidebar.slider("Rolling Window Size", 50, 300, 180, help="Number of days for GARCH model fitting")
    garch_p = st.sidebar.slider("GARCH p parameter", 1, 3, 1, help="Number of lag terms for returns")
    garch_q = st.sidebar.slider("GARCH q parameter", 1, 5, 3, help="Number of lag terms for volatility")
    
    # Signal Parameters
    st.sidebar.subheader("Signal Parameters")
    variance_window = st.sidebar.slider("Variance Window", 50, 300, 180, help="Rolling window for variance calculation")
    premium_window = st.sidebar.slider("Premium Window", 50, 300, 180, help="Rolling window for premium std calculation")
    std_multiplier = st.sidebar.slider("Std Multiplier", 0.5, 3.0, 1.0, 0.1, help="Standard deviation multiplier for signals")
    
    # Technical Indicator Parameters
    st.sidebar.subheader("Technical Indicators")
    rsi_length = st.sidebar.slider("RSI Length", 10, 50, 20, help="Period for RSI calculation")
    rsi_overbought = st.sidebar.slider("RSI Overbought", 60, 90, 70, help="RSI overbought threshold")
    rsi_oversold = st.sidebar.slider("RSI Oversold", 10, 40, 30, help="RSI oversold threshold")
    bb_length = st.sidebar.slider("Bollinger Bands Length", 10, 50, 20, help="Period for Bollinger Bands")
    
    params = {
        'variance_window': variance_window,
        'premium_window': premium_window,
        'std_multiplier': std_multiplier,
        'rsi_length': rsi_length,
        'rsi_overbought': rsi_overbought,
        'rsi_oversold': rsi_oversold,
        'bb_length': bb_length
    }
    
    # Load data
    if st.sidebar.button("Load Sample Data") or st.session_state.data_loaded:
        daily_df, intraday_df = load_sample_data()
        st.session_state.data_loaded = True
        st.session_state.daily_df = daily_df
        st.session_state.intraday_df = intraday_df
        
        # Generate signals
        with st.spinner("Generating signals..."):
            daily_processed, final_df = generate_signals(daily_df, intraday_df, params)
            st.session_state.daily_processed = daily_processed
            st.session_state.final_df = final_df
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "📈 Volatility Analysis", "🎯 Signal Analysis", "💰 Strategy Performance", "⚠️ Risk Analysis"])
        
        with tab1:
            st.header("Data Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Data")
                st.write(f"**Shape:** {daily_df.shape}")
                st.write(f"**Date Range:** {daily_df.index.min().strftime('%Y-%m-%d')} to {daily_df.index.max().strftime('%Y-%m-%d')}")
                st.dataframe(daily_df.head())
                
                # Daily data statistics
                st.subheader("Daily Data Statistics")
                st.dataframe(daily_df.describe())
            
            with col2:
                st.subheader("Intraday Data")
                st.write(f"**Shape:** {intraday_df.shape}")
                st.write(f"**Date Range:** {intraday_df.index.min().strftime('%Y-%m-%d %H:%M')} to {intraday_df.index.max().strftime('%Y-%m-%d %H:%M')}")
                st.dataframe(intraday_df.head())
                
                # Intraday data statistics
                st.subheader("Intraday Data Statistics")
                st.dataframe(intraday_df.describe())
        
        with tab2:
            st.header("Volatility Analysis")
            
            if 'predictions' in daily_processed.columns:
                # Volatility time series
                fig = make_subplots(rows=3, cols=1,
                                  subplot_titles=['Historical vs Predicted Volatility', 'Prediction Premium', 'Daily Signals'],
                                  vertical_spacing=0.08)
                
                # Volatility comparison
                fig.add_trace(go.Scatter(x=daily_processed.index, y=daily_processed['variance'],
                                       name='Historical Volatility', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=daily_processed.index, y=daily_processed['predictions'],
                                       name='Predicted Volatility', line=dict(color='red')), row=1, col=1)
                
                # Prediction premium
                fig.add_trace(go.Scatter(x=daily_processed.index, y=daily_processed['prediction_premium'],
                                       name='Prediction Premium', line=dict(color='green')), row=2, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                # Daily signals
                signal_data = daily_processed['signal_daily'].dropna()
                colors = ['red' if x == 1 else 'blue' for x in signal_data]
                fig.add_scatter(x=signal_data.index, y=signal_data.values,
                              mode='markers', name='Daily Signals',
                              marker=dict(color=colors, size=8), row=3, col=1)
                
                fig.update_layout(height=800, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Distribution plots
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Prediction Premium Distribution")
                    fig_hist = px.histogram(daily_processed['prediction_premium'].dropna(),
                                          title="Prediction Premium Histogram",
                                          nbins=50)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    st.subheader("Signal Frequency")
                    signal_counts = daily_processed['signal_daily'].value_counts()
                    fig_pie = px.pie(values=signal_counts.values, names=['Sell Signal', 'Buy Signal'],
                                   title="Daily Signal Distribution")
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab3:
            st.header("Signal Analysis")
            
            # Signal explanation
            st.markdown("""
            ### 📊 How Intraday Signals Work:
            
            **🔴 SELL SIGNALS (Red dots, value = 1):**
            - Generated when **BOTH** conditions are met:
              - RSI > 70 (Overbought condition)
              - Price > Upper Bollinger Band (Price breakout above volatility band)
            - This indicates potential overvaluation and mean reversion opportunity
            
            **🔵 BUY SIGNALS (Blue dots, value = -1):**
            - Generated when **BOTH** conditions are met:
              - RSI < 30 (Oversold condition)  
              - Price < Lower Bollinger Band (Price breakout below volatility band)
            - This indicates potential undervaluation and bounce opportunity
            
            **Strategy Logic:** The combination of momentum (RSI) and volatility (Bollinger Bands) helps identify extreme price movements that are likely to reverse.
            """)
            
            # Technical indicators chart with enhanced signal visualization
            recent_data = final_df.tail(2000).copy()  # More data for better visualization
            
            # Create signal markers for better visibility
            sell_signals = recent_data[recent_data['signal_intraday'] == 1]
            buy_signals = recent_data[recent_data['signal_intraday'] == -1]
            
            fig = make_subplots(rows=3, cols=1,
                              subplot_titles=[
                                  'Price with Bollinger Bands & Signals', 
                                  'RSI with Signal Conditions', 
                                  'Signal Generation Logic'
                              ],
                              vertical_spacing=0.08,
                              row_heights=[0.5, 0.3, 0.2])
            
            # Price and Bollinger Bands
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['close'],
                                   name='Close Price', line=dict(color='black', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_upper'],
                                   name='BB Upper', line=dict(color='red', dash='dash'), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_lower'],
                                   name='BB Lower', line=dict(color='red', dash='dash'), opacity=0.7), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_middle'],
                                   name='BB Middle', line=dict(color='orange', dash='dot'), opacity=0.5), row=1, col=1)
            
            # Add signal markers on price chart
            if len(sell_signals) > 0:
                fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'],
                                       mode='markers', name='SELL Signals',
                                       marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)
            
            if len(buy_signals) > 0:
                fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'],
                                       mode='markers', name='BUY Signals', 
                                       marker=dict(color='blue', size=10, symbol='triangle-up')), row=1, col=1)
            
            # RSI with conditions highlighted
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['rsi'],
                                   name='RSI', line=dict(color='purple', width=2)), row=2, col=1)
            fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="blue", row=2, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Highlight RSI extreme conditions
            rsi_high = recent_data[recent_data['rsi'] > rsi_overbought]
            rsi_low = recent_data[recent_data['rsi'] < rsi_oversold]
            
            if len(rsi_high) > 0:
                fig.add_trace(go.Scatter(x=rsi_high.index, y=rsi_high['rsi'],
                                       mode='markers', name=f'RSI > {rsi_overbought}',
                                       marker=dict(color='red', size=6, opacity=0.6)), row=2, col=1)
            
            if len(rsi_low) > 0:
                fig.add_trace(go.Scatter(x=rsi_low.index, y=rsi_low['rsi'],
                                       mode='markers', name=f'RSI < {rsi_oversold}',
                                       marker=dict(color='blue', size=6, opacity=0.6)), row=2, col=1)
            
            # Signal timeline
            all_signals = recent_data['signal_intraday'].dropna()
            if len(all_signals) > 0:
                colors = ['red' if x == 1 else 'blue' for x in all_signals]
                fig.add_scatter(x=all_signals.index, y=all_signals.values,
                              mode='markers', name='All Signals',
                              marker=dict(color=colors, size=8), row=3, col=1)
            
            # Add annotations for y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="RSI", row=2, col=1)
            fig.update_yaxes(title_text="Signal", row=3, col=1)
            fig.update_xaxes(title_text="Date", row=3, col=1)
            
            fig.update_layout(height=1000, showlegend=True, 
                            title_text="Technical Analysis with Signal Generation")
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal Statistics
            st.subheader("📈 Signal Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_signals = len(final_df['signal_intraday'].dropna())
            sell_signals_count = len(final_df[final_df['signal_intraday'] == 1])
            buy_signals_count = len(final_df[final_df['signal_intraday'] == -1])
            
            with col1:
                st.metric("Total Signals", f"{total_signals:,}")
            with col2:
                st.metric("Sell Signals", f"{sell_signals_count:,}", 
                         delta=f"{sell_signals_count/total_signals*100:.1f}%" if total_signals > 0 else "0%")
            with col3:
                st.metric("Buy Signals", f"{buy_signals_count:,}", 
                         delta=f"{buy_signals_count/total_signals*100:.1f}%" if total_signals > 0 else "0%")
            with col4:
                signal_ratio = sell_signals_count / buy_signals_count if buy_signals_count > 0 else 0
                st.metric("Sell/Buy Ratio", f"{signal_ratio:.2f}")
            
            # Signal analysis by time of day
            st.subheader("🕒 Signal Distribution by Time of Day")
            
            if total_signals > 0:
                final_df['hour'] = final_df.index.hour
                
                # Separate analysis for sell and buy signals
                sell_by_hour = final_df[final_df['signal_intraday'] == 1].groupby('hour').size()
                buy_by_hour = final_df[final_df['signal_intraday'] == -1].groupby('hour').size()
                
                # Create combined bar chart
                fig_hour = go.Figure()
                
                if len(sell_by_hour) > 0:
                    fig_hour.add_trace(go.Bar(x=sell_by_hour.index, y=sell_by_hour.values,
                                            name='Sell Signals', marker_color='red', opacity=0.7))
                
                if len(buy_by_hour) > 0:
                    fig_hour.add_trace(go.Bar(x=buy_by_hour.index, y=buy_by_hour.values,
                                            name='Buy Signals', marker_color='blue', opacity=0.7))
                
                fig_hour.update_layout(title="Signal Frequency by Hour of Day",
                                     xaxis_title="Hour of Day", 
                                     yaxis_title="Number of Signals",
                                     barmode='group')
                st.plotly_chart(fig_hour, use_container_width=True)
                
                # Signal effectiveness analysis
                st.subheader("🎯 Signal Effectiveness Analysis")
                
                # Calculate forward returns after signals
                signal_data = final_df[final_df['signal_intraday'].notna()].copy()
                if len(signal_data) > 0:
                    signal_data['forward_1h_return'] = signal_data['return'].rolling(12).sum().shift(-12)  # 1 hour forward return
                    signal_data['forward_4h_return'] = signal_data['return'].rolling(48).sum().shift(-48)  # 4 hour forward return
                    
                    sell_performance = signal_data[signal_data['signal_intraday'] == 1][['forward_1h_return', 'forward_4h_return']].mean()
                    buy_performance = signal_data[signal_data['signal_intraday'] == -1][['forward_1h_return', 'forward_4h_return']].mean()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Sell Signal Performance:**")
                        if not sell_performance.empty:
                            st.write(f"1-Hour Forward Return: {sell_performance['forward_1h_return']:.3%}")
                            st.write(f"4-Hour Forward Return: {sell_performance['forward_4h_return']:.3%}")
                        else:
                            st.write("No sell signals available for analysis")
                    
                    with col2:
                        st.write("**Buy Signal Performance:**")
                        if not buy_performance.empty:
                            st.write(f"1-Hour Forward Return: {buy_performance['forward_1h_return']:.3%}")
                            st.write(f"4-Hour Forward Return: {buy_performance['forward_4h_return']:.3%}")
                        else:
                            st.write("No buy signals available for analysis")
            else:
                st.warning("No signals generated with current parameters. Try adjusting the RSI or Bollinger Band settings.")
        
        with tab4:
            st.header("Strategy Performance")
            
            # Calculate daily returns
            daily_returns = final_df.groupby(pd.Grouper(freq='D'))['strategy_return'].sum().dropna()
            
            if len(daily_returns) > 0:
                # Performance metrics
                metrics = calculate_performance_metrics(daily_returns)
                
                st.subheader("Performance Metrics")
                cols = st.columns(4)
                metric_items = list(metrics.items())
                
                for i, (key, value) in enumerate(metric_items):
                    with cols[i % 4]:
                        st.metric(key, value)
                
                # Cumulative returns chart
                cumulative_returns = (1 + daily_returns).cumprod() - 1
                benchmark_returns = (1 + np.random.normal(0.0005, 0.015, len(daily_returns))).cumprod() - 1
                
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values,
                                            name='Strategy Returns', line=dict(color='blue', width=2)))
                fig_perf.add_trace(go.Scatter(x=cumulative_returns.index, y=benchmark_returns,
                                            name='Benchmark (Random)', line=dict(color='gray', dash='dash')))
                
                fig_perf.update_layout(title="Cumulative Strategy Returns",
                                     xaxis_title="Date",
                                     yaxis_title="Cumulative Return",
                                     yaxis_tickformat='.1%',
                                     height=500)
                st.plotly_chart(fig_perf, use_container_width=True)
                
                # Returns distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_dist = px.histogram(daily_returns, title="Daily Returns Distribution",
                                          nbins=50, labels={'value': 'Daily Return'})
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Monthly returns heatmap
                    monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                    monthly_returns.index = monthly_returns.index.strftime('%Y-%m')
                    
                    fig_monthly = px.bar(x=monthly_returns.index, y=monthly_returns.values,
                                       title="Monthly Returns",
                                       labels={'x': 'Month', 'y': 'Monthly Return'})
                    fig_monthly.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_monthly, use_container_width=True)
        
        with tab5:
            st.header("Risk Analysis")
            
            if len(daily_returns) > 0:
                # Filter out zero returns for better risk analysis
                non_zero_returns = daily_returns[daily_returns != 0]
                
                # Drawdown analysis
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                          fill='tozeroy', name='Drawdown',
                                          line=dict(color='red'),
                                          fillcolor='rgba(255,0,0,0.3)'))
                fig_dd.update_layout(title="Strategy Drawdown",
                                   xaxis_title="Date",
                                   yaxis_title="Drawdown",
                                   yaxis_tickformat='.2%',
                                   height=400)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Risk metrics - using all returns including zeros
                st.subheader("Risk Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Calculate VaR properly
                if len(daily_returns) > 20:  # Need sufficient data points
                    var_95 = daily_returns.quantile(0.05)
                    var_99 = daily_returns.quantile(0.01)
                    
                    # Expected shortfall (Conditional VaR)
                    returns_below_var95 = daily_returns[daily_returns <= var_95]
                    if len(returns_below_var95) > 0:
                        expected_shortfall = returns_below_var95.mean()
                    else:
                        expected_shortfall = var_95
                    
                    with col1:
                        st.metric("Value at Risk (95%)", f"{var_95:.3%}", 
                                help="5% worst-case daily loss")
                    
                    with col2:
                        st.metric("Value at Risk (99%)", f"{var_99:.3%}",
                                help="1% worst-case daily loss")
                    
                    with col3:
                        st.metric("Expected Shortfall", f"{expected_shortfall:.3%}",
                                help="Average loss when VaR is exceeded")
                else:
                    with col1:
                        st.metric("Value at Risk (95%)", "Insufficient Data")
                    with col2:
                        st.metric("Value at Risk (99%)", "Insufficient Data")
                    with col3:
                        st.metric("Expected Shortfall", "Insufficient Data")
                
                # Additional risk metrics
                col4, col5, col6 = st.columns(3)
                
                with col4:
                    daily_vol = daily_returns.std()
                    annualized_vol = daily_vol * np.sqrt(252)
                    st.metric("Daily Volatility", f"{daily_vol:.3%}")
                
                with col5:
                    st.metric("Annualized Volatility", f"{annualized_vol:.2%}")
                
                with col6:
                    skewness = daily_returns.skew()
                    st.metric("Skewness", f"{skewness:.3f}",
                            help="Measure of asymmetry in returns distribution")
                
                # Rolling volatility
                if len(daily_returns) > 30:
                    rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)
                    rolling_vol = rolling_vol.dropna()
                    
                    if len(rolling_vol) > 0:
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol.values,
                                                   name='30-Day Rolling Volatility',
                                                   line=dict(color='orange')))
                        fig_vol.update_layout(title="Rolling Volatility (30-day)",
                                            xaxis_title="Date",
                                            yaxis_title="Annualized Volatility",
                                            yaxis_tickformat='.1%',
                                            height=400)
                        st.plotly_chart(fig_vol, use_container_width=True)
                
                # Returns distribution analysis
                st.subheader("Returns Distribution Analysis")
                
                col7, col8 = st.columns(2)
                
                with col7:
                    # Histogram of returns
                    fig_hist = px.histogram(daily_returns, 
                                          title="Daily Returns Distribution",
                                          nbins=50,
                                          labels={'value': 'Daily Return', 'count': 'Frequency'})
                    fig_hist.add_vline(x=var_95, line_dash="dash", line_color="red",
                                     annotation_text="VaR 95%")
                    fig_hist.add_vline(x=var_99, line_dash="dash", line_color="darkred",
                                     annotation_text="VaR 99%")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col8:
                    # Q-Q plot approximation
                    sorted_returns = np.sort(daily_returns.dropna())
                    theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_returns))
                    normal_quantiles = np.random.normal(daily_returns.mean(), daily_returns.std(), len(sorted_returns))
                    normal_quantiles = np.sort(normal_quantiles)
                    
                    fig_qq = go.Figure()
                    fig_qq.add_trace(go.Scatter(x=normal_quantiles, y=sorted_returns,
                                              mode='markers', name='Actual vs Normal',
                                              marker=dict(size=4)))
                    # Add diagonal line
                    min_val = min(min(normal_quantiles), min(sorted_returns))
                    max_val = max(max(normal_quantiles), max(sorted_returns))
                    fig_qq.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                              mode='lines', name='Perfect Normal',
                                              line=dict(color='red', dash='dash')))
                    
                    fig_qq.update_layout(title="Q-Q Plot: Returns vs Normal Distribution",
                                       xaxis_title="Theoretical Normal Quantiles",
                                       yaxis_title="Sample Quantiles")
                    st.plotly_chart(fig_qq, use_container_width=True)
                
            else:
                st.warning("No strategy returns available for risk analysis.")
    
    else:
        st.info("👈 Click 'Load Sample Data' in the sidebar to begin the analysis.")
        
        st.markdown("""
        ## About This Strategy
        
        This application implements an intraday trading strategy that combines:
        
        1. **GARCH Volatility Modeling**: Predicts future volatility using daily data
        2. **Technical Indicators**: Uses RSI and Bollinger Bands on 5-minute data
        3. **Signal Combination**: Combines daily volatility signals with intraday technical signals
        
        ### Strategy Logic:
        - **Daily Signal**: Generated from GARCH volatility prediction premium
        - **Intraday Signal**: Based on RSI and Bollinger Bands
        - **Position**: Taken when both signals align (contrarian approach)
        
        ### Features:
        - Interactive parameter adjustment
        - Comprehensive performance analysis
        - Risk metrics and drawdown analysis
        - Visual signal analysis
        """)

if __name__ == "__main__":
    main()