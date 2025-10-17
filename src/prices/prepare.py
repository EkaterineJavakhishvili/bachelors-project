import pandas as pd
from src.config import RAW_PRICES, INT_PRICES, TICKER


# From CSV sorts data chronologically, drops colums with missing values
# Write new parsed and modified dataset to .parquet file
def prepare_prices(ticker: str = TICKER):
    src = RAW_PRICES / f"{ticker}.csv"
    df = pd.read_csv(src, parse_dates=["date"]).set_index("date").sort_index()

    df = df.dropna()
    out = INT_PRICES / f"{ticker}.parquet"
    df.to_parquet(out)
    print(f"{ticker} cleaned")


if __name__ == "__main__":
    prepare_prices()
