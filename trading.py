# Intra-day trading signals
# Python script converted from trading.ipynb

# 1. Load Simulated Daily and Simulated 5-minute data
import matplotlib as plt
from arch import arch_model
import pandas_ta
import pandas as pd
import numpy as np
import os

data_folder = '/Users/ayushpetwal/Desktop/JBooks'

daily_df = pd.read_csv(os.path.join(data_folder, 'simulated_daily_data.csv'))

daily_df = daily_df.drop('Unnamed: 7', axis=1)

daily_df['Date'] = pd.to_datetime(daily_df['Date'])

daily_df = daily_df.set_index('Date')

intraday_5min_df = pd.read_csv(os.path.join(data_folder, 'simulated_5min_data.csv'))

intraday_5min_df = intraday_5min_df.drop('Unnamed: 6', axis=1)

intraday_5min_df['datetime'] = pd.to_datetime(intraday_5min_df['datetime'])

intraday_5min_df = intraday_5min_df.set_index('datetime')

intraday_5min_df['date'] = pd.to_datetime(intraday_5min_df.index.date)

print("Data loaded successfully")
print(f"Daily data shape: {daily_df.shape}")
print(f"Intraday data shape: {intraday_5min_df.shape}")

# 2. Define function to fit GARCH model and predict 1-day ahead volatility in a rolling window
daily_df['log_ret'] = np.log(daily_df['Adj Close']).diff()

daily_df['variance'] = daily_df['log_ret'].rolling(180).var()

daily_df = daily_df['2020':]

def predict_volatility(x):
    best_model = arch_model(y=x,
                            p=1,
                            q=3).fit(update_freq=5,
                                   disp='off')
    
    variance_forecast = best_model.forecast(horizon=1).variance.iloc[-1,0]

    print(x.index[-1])
    
    return variance_forecast

daily_df['predictions'] = daily_df['log_ret'].rolling(180).apply(lambda x: predict_volatility(x))

daily_df = daily_df.dropna()

print("GARCH volatility predictions completed")
print(f"Daily data shape after predictions: {daily_df.shape}")

# 3. Calculate prediction premium and form a daily signal from it
daily_df[['variance', 'predictions']].plot()

daily_df['prediction_premium'] = (daily_df['predictions']-daily_df['variance'])/daily_df['variance']

daily_df['premium_std'] = daily_df['prediction_premium'].rolling(180).std()

daily_df['signal_daily'] = daily_df.apply(lambda x: 1 if (x['prediction_premium']>x['premium_std'])
                                         else (-1 if (x['prediction_premium']<x['premium_std']*-1) else np.nan),
                                         axis=1)

daily_df['signal_daily'] = daily_df['signal_daily'].shift()

print("Daily signals generated")

import matplotlib.pyplot as plt

plt.style.use('ggplot')

daily_df['signal_daily'].plot(kind='hist')

plt.show()

# 4. Merge with intraday data and calculate intraday indicators to form the intraday signal
final_df = intraday_5min_df.reset_index()\
                            .merge(daily_df[['signal_daily']].reset_index(),
                                   left_on='date',
                                   right_on='Date')\
                            .drop(['date','Date'], axis=1)\
                            .set_index('datetime')

final_df['rsi'] = pandas_ta.rsi(close=final_df['close'],
                            length=20)

final_df['lband'] = pandas_ta.bbands(close=final_df['close'],
                                   length=20).iloc[:,0]

final_df['uband'] = pandas_ta.bbands(close=final_df['close'],
                                   length=20).iloc[:,2]

final_df['signal_intraday'] = final_df.apply(lambda x: 1 if (x['rsi']>70)&
                                                        (x['close']>x['uband'])
                                                 else (-1 if (x['rsi']<30)&
                                                        (x['close']<x['lband']) else np.nan),
                                                 axis=1)

final_df['return'] = np.log(final_df['close']).diff()

print("Intraday indicators and signals calculated")
print(f"Final dataframe shape: {final_df.shape}")

# 5. Generate the position entry and hold until the end of the day
final_df['return_sign'] = final_df.apply(lambda x: -1 if (x['signal_daily']==1)&(x['signal_intraday']==1)
                                        else (1 if (x['signal_daily']==-1)&(x['signal_intraday']==-1) else np.nan),
                                        axis=1)

final_df['return_sign'] = final_df.groupby(pd.Grouper(freq='D'))['return_sign']\
                                  .transform(lambda x: x.ffill())

final_df['forward_return'] = final_df['return'].shift(-1)

final_df['strategy_return'] = final_df['forward_return']*final_df['return_sign']

daily_return_df = final_df.groupby(pd.Grouper(freq='D'))['strategy_return'].sum()

print("Position entries and returns calculated")

# 6. Calculate final strategy returns
import matplotlib.ticker as mtick

strategy_cumulative_return = np.exp(np.log1p(daily_return_df).cumsum()).sub(1)

strategy_cumulative_return.plot(figsize=(16,6))

plt.title('Intraday Strategy Returns')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()

print("Strategy analysis completed!")
print(f"Final cumulative return: {strategy_cumulative_return.iloc[-1]:.2%}")