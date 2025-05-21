
import yfinance as yf
import pandas as pd
from pathlib import Path # Added

def download_stock_data(ticker: str, start: str, end: str):
    stock_data = yf.download(ticker, start=start, end=end)
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data.reset_index(inplace=True)
    return stock_data

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent
    output_dir = BASE_DIR / "data"
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "aapl_stock_data.csv"
    
    stock_data = download_stock_data('AAPL', '2010-01-01', '2023-01-01')
    stock_data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
