import requests
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime

def format_number(number):
    return f"{number:,}"

def get_stock_list(filter_US_only=True):
  place_to_save_list = "stocks/all_stocks_iex.json"
  output = requests.get("https://cloud.iexapis.com/beta/ref-data/symbols?token=pk_467b3fec4bbd430ebc812b3e8c891f6e")
  data = output.json()
  # save this
  if filter_US_only:
      stocks = [item['symbol'] for item in data if "-" not in item['symbol']]
  else:
      stocks = [item['symbol'] for item in data]
  print("len: ", len(stocks))
  if not os.path.exists(os.path.split(place_to_save_list)[0]):
    os.makedirs(os.path.split(place_to_save_list)[0])
  with open(place_to_save_list, "w") as f:
    json.dump(data, f)
  return stocks

def convert_rating(rating):
    if rating < -50:
        return"*This suggests a strong sell and would be a poor investment decision"
    if rating >= -50 and rating < -25:

        return "*This suggests a sell"
    if rating >= -25 and rating < 0:
        return "*This suggests a marginal sell"

    if rating >= 0 and rating < 25:
        return "*This suggest a marginal buy"
    if rating >= 25 and rating < 50:
        return "*This suggest a buy signal"
    if rating >= 50 and rating < 100:
        return "*This suggest a significant buy opportunity"

    return "* no definite recommendation made"

def convert_df(df):
    return df.to_csv().encode('utf-8')

def give_cum_return(df):
    cum_return = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    return cum_return

def give_log_std(df):
    log_price= np.log(df)
    mean= np.mean(log_price)
    return np.std(log_price)

def give_cum_return(df):
    cum_return = (df.iloc[-1] - df.iloc[0]) / df.iloc[0]
    return cum_return

def give_std(df):
    return np.std(df)

## for functions page

def string_to_time(string, _format="%d/%m/%Y"):
    return datetime.strptime(string, _format)

def time_to_string(time, _format="%d/%m/%Y"):
    return datetime.strftime(time, _format)

def filters_date(df, start_date=None, end_date=None):
    if start_date==None:
        start_date = string_to_time("01/01/1900")
    else:
        start_date = string_to_time(start_date)
    if end_date==None:
        end_date   = string_to_time("01/01/2100")
    else:
        end_date   = string_to_time(end_date)
    start_date_combined = datetime.combine(start_date, datetime.min.time())
    end_date_combined = datetime.combine(end_date, datetime.min.time())
    datetimes = np.array([datetime.strptime(date, "%Y-%m-%d") for date in df["Date"]])
    indices = np.logical_and(datetimes > start_date_combined, datetimes < end_date_combined)
    return df.loc[indices]

