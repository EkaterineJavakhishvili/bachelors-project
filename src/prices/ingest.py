import pandas as pd
import yfinance as yf
from src.config import RAW_PRICES, TICKER, START, END


# Pulls the prices from Yahoo Finance
def get_prices(ticker: str = TICKER, start: str = START, end: str = END):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # To avoid returning multi-level or MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(f"No data for {ticker}")

    df.index.name = "date"
    out = RAW_PRICES / f"{ticker}.csv"
    df.to_csv(out)
    print(f"{ticker} Saved")


if __name__ == "__main__":
    get_prices()
