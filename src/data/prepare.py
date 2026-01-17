import argparse
import pandas as pd
from pathlib import Path
from src.config import RAW_PRICES, PROC_PRICES, TICKER
from src.utils import standardize_columns


def prepare_data(ticker: str = TICKER) -> None:
    """
    Cleans and standardizes raw stock data.

    Transformatons:
    1. standardize column names (title case)
    2. convert date to datetime objects
    3. sort chronologically
    4. remove missing values
    5. save as optimized parquet file

    Args:
        ticker (str): The stock symbol.
    """
    print(f"\n🧹 CLEANING DATA FOR {ticker}...")

    try:
        # 1. load raw data
        input_path = RAW_PRICES / f"{ticker}.csv"

        if not input_path.exists():
            raise FileNotFoundError(
                f"❌ Raw file not found: {input_path}\nRun 'ingest.py' first."
            )

        df = pd.read_csv(input_path)

        # 2. standardize columns
        # fixes 'date' -> 'Date', 'close' -> 'Close', etc
        df = standardize_columns(df)

        # 3. parse dates
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
            df = df.set_index("Date")
        else:
            raise ValueError(f"Column 'Date' missing in {ticker}.csv")

        # 4. clean and sort
        df = df.sort_index()
        initial_len = len(df)
        df = df.dropna()
        dropped_len = initial_len - len(df)

        if dropped_len > 0:
            print(f"⚠️ Dropped {dropped_len} rows with missing values.")

        # 5. save to interim
        PROC_PRICES.mkdir(parents=True, exist_ok=True)

        out_path = PROC_PRICES / f"{ticker}.parquet"
        df.to_parquet(out_path)

        print(f"✅ SUCCESS: Cleaned data saved to {out_path}")
        print(f"   Range: {df.index.min()} to {df.index.max()}")

    except Exception as e:
        print(f"❌ ERROR in Data Preparation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean raw stock data.")

    parser.add_argument(
        "--ticker", type=str, default=TICKER, help=f"Stock symbol (default: {TICKER})"
    )

    args = parser.parse_args()

    prepare_data(ticker=args.ticker)
