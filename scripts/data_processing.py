import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import pickle

os.makedirs("../data/raw", exist_ok=True)
os.makedirs("../data/processed", exist_ok=True)

def download_data(etfs, start_date, end_date):
    df = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    df.dropna(inplace=True)
    df.to_csv("../data/raw/etf_prices.csv", index=True)
    return df

def calculate_log_returns(df):
    log_returns = np.log(df / df.shift(1)).dropna()
    log_returns.to_csv("../data/processed/log_returns.csv", index=True)
    return log_returns

def save_split_data(log_returns, train_size=0.7, val_size=0.15):
    total = len(log_returns)
    train_end = int(total * train_size)
    val_end = train_end + int(total * val_size)

    train_data = log_returns.iloc[:train_end]
    val_data = log_returns.iloc[train_end:val_end]
    test_data = log_returns.iloc[val_end:]

    train_data.to_csv("../data/processed/train_data.csv", index=True)
    val_data.to_csv("../data/processed/val_data.csv", index=True)
    test_data.to_csv("../data/processed/test_data.csv", index=True)

    return train_data, val_data, test_data

def scale_data(data, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler

# Usage Example
# etfs = ['SPY', 'TLT', 'SHY', 'GLD', 'DBO']
# df = download_data(etfs, '2018-01-01', '2022-12-30')
# log_returns = calculate_log_returns(df)
# train_data, val_data, test_data = save_split_data(log_returns)
