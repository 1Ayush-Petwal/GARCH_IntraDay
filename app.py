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
    final_df['bb_lower'] = bb.iloc[:, 0]
    final_df['bb_upper'] = bb.iloc[:, 2]
    final_df['bb_middle'] = bb.iloc[:, 1]
    
    # Intraday signal
    final_df['signal_intraday'] = final_df.apply(
        lambda x: 1 if (x['rsi'] > params['rsi_overbought']) & (x['close'] > x['bb_upper'])
        else (-1 if (x['rsi'] < params['rsi_oversold']) & (x['close'] < x['bb_lower']) else np.nan),
        axis=1
    )
    
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
            
            # Technical indicators chart
            recent_data = final_df.tail(1000)  # Last 1000 observations for better visualization
            
            fig = make_subplots(rows=3, cols=1,
                              subplot_titles=['Price with Bollinger Bands', 'RSI', 'Intraday Signals'],
                              vertical_spacing=0.08)
            
            # Price and Bollinger Bands
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['close'],
                                   name='Close Price', line=dict(color='black')), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_upper'],
                                   name='BB Upper', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_lower'],
                                   name='BB Lower', line=dict(color='red', dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['bb_middle'],
                                   name='BB Middle', line=dict(color='orange', dash='dot')), row=1, col=1)
            
            # RSI
            fig.add_trace(go.Scatter(x=recent_data.index, y=recent_data['rsi'],
                                   name='RSI', line=dict(color='purple')), row=2, col=1)
            fig.add_hline(y=rsi_overbought, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=rsi_oversold, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Intraday signals
            intraday_signals = recent_data['signal_intraday'].dropna()
            if len(intraday_signals) > 0:
                colors = ['red' if x == 1 else 'blue' for x in intraday_signals]
                fig.add_scatter(x=intraday_signals.index, y=intraday_signals.values,
                              mode='markers', name='Intraday Signals',
                              marker=dict(color=colors, size=6), row=3, col=1)
            
            fig.update_layout(height=900, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal analysis by time of day
            st.subheader("Signal Analysis by Time of Day")
            final_df['hour'] = final_df.index.hour
            signal_by_hour = final_df.groupby('hour')['signal_intraday'].count()
            
            fig_hour = px.bar(x=signal_by_hour.index, y=signal_by_hour.values,
                             title="Intraday Signal Frequency by Hour",
                             labels={'x': 'Hour of Day', 'y': 'Number of Signals'})
            st.plotly_chart(fig_hour, use_container_width=True)
        
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
                # Drawdown analysis
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values,
                                          fill='tonexty', name='Drawdown',
                                          line=dict(color='red')))
                fig_dd.update_layout(title="Strategy Drawdown",
                                   xaxis_title="Date",
                                   yaxis_title="Drawdown",
                                   yaxis_tickformat='.1%',
                                   height=400)
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Risk metrics
                st.subheader("Risk Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    var_95 = daily_returns.quantile(0.05)
                    st.metric("Value at Risk (95%)", f"{var_95:.2%}")
                
                with col2:
                    var_99 = daily_returns.quantile(0.01)
                    st.metric("Value at Risk (99%)", f"{var_99:.2%}")
                
                with col3:
                    expected_shortfall = daily_returns[daily_returns <= var_95].mean()
                    st.metric("Expected Shortfall", f"{expected_shortfall:.2%}")
                
                # Rolling volatility
                rolling_vol = daily_returns.rolling(30).std() * np.sqrt(252)
                
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