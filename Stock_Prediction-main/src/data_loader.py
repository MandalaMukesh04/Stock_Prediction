import yfinance as yf
import pandas as pd

def load_data(stock_symbol):
    data = yf.download(stock_symbol, period="1y")  # Fetch data for the last 1 year
    return pd.DataFrame(data)  # Ensure it returns a DataFrame
