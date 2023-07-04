from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import time
import math


"""
Converts a date string to a integer unix time.
"""
def datestring_to_unix(date_string):
    return int(time.mktime(datetime.strptime(date_string, '%Y-%m-%d').timetuple()))


"""
Takes in a ticker symbol, start date, and number of days to count back.
Gets data from API ranging from 20 days before that date to the day after that date (for later signal purposes).
"""
def get_stock_data(symbol, start_date, countback):
    start_date = datestring_to_unix(start_date)
    base_url = f"https://api.marketdata.app/v1/stocks/candles/D/{symbol}?to={start_date}&countback={countback}"
    response = requests.get(base_url)
    data = response.json()
    
    return data


"""
Takes in a dataframe with a closing price column.
Returns a dataframe with a fast avg, slow avg, MACD, signal, and histogram columns.
"""
def add_MACD_cols(df, n_fast=12, n_slow=26, n_smooth=9):
    closing_prices = df["c"]
    fast_EMA = closing_prices.ewm(span=n_fast, min_periods=n_slow).mean()
    slow_EMA = closing_prices.ewm(span=n_slow, min_periods=n_slow).mean()
    fast_col = pd.Series(fast_EMA, name = "fastEMA")
    slow_col = pd.Series(slow_EMA, name = "slowEMA")
    MACD_col = pd.Series(fast_EMA - slow_EMA, name = 'MACD')
    MACDsignal = pd.Series(MACD_col.ewm(span=n_smooth, min_periods=n_smooth).mean(), name='MACDsig')
    MACDhistogram = pd.Series(MACD_col - MACDsignal, name = 'MACDhist')
    df = df.join(fast_col)
    df = df.join(slow_col)
    df = df.join(MACD_col)
    df = df.join(MACDsignal)
    df = df.join(MACDhistogram)
    return df


"""
Takes in a dataframe with a MACD and signal line columns.
Returns a dataframe with a buy signals column and a sell signals column, with non signal periods filled with NaN.
"""
def add_signal_cols(df):
    buy = []
    sell = []
    begin_above = None
    
    first_index = df['MACDhist'].notna().idxmax()
    if df['MACDhist'][first_index] > 0:
        begin_above = True
    else:
        begin_above = False

    if begin_above:
        state = "above"
        for i in range(len(df)):
            if df["MACD"][i] > df["MACDsig"][i] and state == "below":
                buy.append(df["c"][i])
                sell.append(np.NaN)
                state = "above"
            elif df["MACD"][i] < df["MACDsig"][i] and state == "above":
                sell.append(df["c"][i])
                buy.append(np.NaN)
                state = "below"
            else:
                buy.append(np.NaN)
                sell.append(np.NaN)

    else:
        state = "below"
        for i in range(len(df)):
            if df["MACD"][i] > df["MACDsig"][i] and state == "below":
                buy.append(df["c"][i])
                sell.append(np.NaN)
                state = "above"
            elif df["MACD"][i] < df["MACDsig"][i] and state == "above":
                sell.append(df["c"][i])
                buy.append(np.NaN)
                state = "below"
            else:
                buy.append(np.NaN)
                sell.append(np.NaN)

    buy_col = pd.Series(buy, name = "buy_sig")
    sell_col = pd.Series(sell, name = "sell_sig")

    df = df.join(buy_col)
    df = df.join(sell_col)

    return df



def plot_signals(df, ticker):
    # plot price
    plt.figure(figsize=(15,5))
    plt.plot(df['t'], df['c'])
    plt.title('Price chart (Adj Close) ' + str(ticker))
    plt.show()

    # plot  values and significant levels
    plt.figure(figsize=(15,5))
    plt.title('Bollinger Bands chart ' + str(ticker))
    plt.plot(df['t'], df['h'], label='High', alpha=0.2)
    plt.plot(df['t'], df['l'], label='Low', alpha=0.2)
    plt.plot(df['t'], df['c'], label='Adj Close', color='blue', alpha=0.3)

    plt.scatter(df['t'], df['buy_sig'], label='Buy', marker='^')
    plt.scatter(df['t'], df['sell_sig'], label='Sell', marker='v')

    plt.legend()

    plt.show()
       
    plt.figure(figsize=(15,5))
    plt.title('MACD chart ' + str(ticker))
    plt.plot(df['t'], df['MACD'].fillna(0))
    plt.plot(df['t'], df['MACDsig'].fillna(0))
    plt.plot(df['t'], df['MACDhist'].fillna(0))
    plt.bar(df['t'], df['MACDhist'].fillna(0), width=0.5, snap=False)
    
    return None


"""
Takes in a dataframe of previous data, a ticker (should be the same as the previous data), and a new date.
Returns a dataframe that is the same as the previous dataframe with the new information added to the bottom.
"""
def add_new_day_row(df, ticker, date):
    stock_data = get_stock_data(ticker, date, 1)
    new_row = [stock_data['s'][0], stock_data['o'][0], stock_data['c'][0], stock_data['h'][0], 
               stock_data['l'][0], stock_data['v'][0], stock_data['t'][0]]
    new_row_df = pd.DataFrame(stock_data)

    previous_fast_EMA = df['fastEMA'][len(df)-1]
    previous_slow_EMA = df['slowEMA'][len(df)-1]

    new_fast_EMA = (new_row_df['c'][0] - previous_fast_EMA) * (2 / 13) + previous_fast_EMA
    new_slow_EMA = (new_row_df['c'][0] - previous_slow_EMA) * (2 / 27) + previous_slow_EMA
    new_row.append(new_fast_EMA)
    new_row.append(new_slow_EMA)

    new_MACD = new_fast_EMA - new_slow_EMA
    new_row.append(new_MACD)

    new_signal = (new_MACD - df['MACDsig'][len(df)-1]) * (2 / 10) + df['MACDsig'][len(df)-1]
    new_row.append(new_signal)

    new_hist = new_MACD - new_signal
    new_row.append(new_hist)

    previous_MACD = df['MACD'][len(df)-1]
    previous_signal = df['MACDsig'][len(df)-1]

    if previous_MACD - previous_signal > 0:
        state = "above"
    else:
        state = "below"
    
    if new_MACD > new_signal and state == "below":
        new_row.append(stock_data['c'][0])
        new_row.append(np.NaN)
    elif new_MACD < new_signal and state == "above":
        new_row.append(np.NaN)
        new_row.append(stock_data['c'][0])
    else:
        new_row.append(np.NaN)
        new_row.append(np.NaN)

    new_row = pd.Series(new_row, index = df.columns)
    df = df._append(new_row, ignore_index = True)

    return df


"""
Takes in a ticker, a starting amount of money, a start date, and an end date.
Returns the end amount of money from using the MACD strategy during those dates with that amount of starting money.
"""
def MACD_historical_backtest(ticker, starting_amount, start_date, countback):
    data = get_stock_data(ticker, start_date, countback)
    df = pd.DataFrame(data)
    df = add_MACD_cols(df, 12, 26, 9)
    df = add_signal_cols(df)

    stock_amount = starting_amount / 2
    liquid_amount = starting_amount - stock_amount
    previous_close = df['c'][0]
    for index, row in df.iterrows():
        stock_amount = (row['c'] / previous_close) * stock_amount
        previous_close = row['c']

        print(f"Amount invested in stocks: {stock_amount} and amount liquid: {liquid_amount} on day {index}")

        if not math.isnan(row['buy_sig']):
            old_stock_amount = stock_amount
            stock_amount, liquid_amount = stock_amount + liquid_amount * 0.1, liquid_amount * 0.9
            print(f"Bought {stock_amount - old_stock_amount} of {ticker} on {row['t']} / day {index}")
        elif not math.isnan(row['sell_sig']):
            old_stock_amount = stock_amount
            stock_amount, liquid_amount = stock_amount * 0.9, liquid_amount + stock_amount * 0.1
            print(f"Sold {old_stock_amount - stock_amount} of {ticker} on {row['t']} / day {index}")

    return stock_amount, liquid_amount, stock_amount + liquid_amount



if __name__ == '__main__':
    result = MACD_historical_backtest("AAPL", 100000, "2022-06-22", 1000)
    print(f"Ended with {result[0]} in stocks, {result[1]} in cash, and {result[2]} in total.")



