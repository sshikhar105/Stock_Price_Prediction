
import yfinance as yf
import pandas as pd

def download_stock_data(ticker: str, start: str, end: str):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data.reset_index(inplace=True)
    return stock_data

if __name__ == "__main__":
    stock_data = download_stock_data('AAPL', '2010-01-01', '2023-01-01')
    stock_data.to_csv('../data/aapl_stock_data.csv', index=False)
