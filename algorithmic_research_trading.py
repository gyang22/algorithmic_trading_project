# -*- coding: utf-8 -*-
"""algorithmic_research_trading.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rhmOzeY0OTT-0HNC1X_ducotShMKoiZm
"""

import requests
import numpy as np
import time
from datetime import datetime, timedelta


"""
Converts a unix timestamp to a YYYY-MM-DD string format.
"""
def unix_to_datestring(time_number):
    return datetime.utcfromtimestamp(time_number).strftime('%Y-%m-%d')


"""
Converts a YYYY-MM-DD string to a unix timestamp.
"""
def datestring_to_unix(date_string):
    return int(time.mktime(datetime.strptime(date_string, "%Y-%m-%d").timetuple()))


"""
Takes in a stock ticker, start date either in YYYY-MM-DD format or unix timestamp, and a number of days n to count backward.
Returns the win/loss of each day in that period (close - open) as a numpy array.
"""
def get_n_day_win_loss(stock_ticker, start_date, n):
    if type(start_date) == str:
        start_date = datestring_to_unix(start_date)
    behind_date = int(time.mktime((datetime.fromtimestamp(start_date) - timedelta(days=n-1)).timetuple()))
    url = f"https://api.marketdata.app/v1/stocks/candles/D/{stock_ticker}?from={behind_date}&to={start_date}"
    response = requests.request("GET", url)
    data = response.json()
    win_loss = np.array(data["c"]) - np.array(data["o"])
    print(f"Number of results: {len(win_loss)}")
    return win_loss


"""
Takes in a win/loss array.
Returns either "buy" or "sell" based on the absolute number of gains to losses over the period.
"""
def absolute_number_strategy(win_loss_array):
    if type(win_loss_array) != np.array:
        raise Exception("Please enter valid numpy array.")
    wins = len([x for x in win_loss_array if x > 0])
    losses = len([x for x in win_loss_array if x <= 0])
    return "buy" if wins > losses else "sell"


"""
Takes in a win/loss array.
Returns either "buy" or "sell" based on the total aggregate profit of the period.
"""
def aggregate_sum_strategy(win_loss_array):
    if type(win_loss_array) != np.array:
        raise Exception("Please enter valid numpy array.")
    return "buy" if sum(win_loss_array) > 0 else "sell"

start_time = time.time()

np.set_printoptions(suppress = True)
print(get_n_day_win_loss("AAPL", "2020-12-03", 20))

end_time = time.time()

print(f"Process took {end_time-start_time} seconds.")