import argparse
import pandas as pd
import yfinance as yf
from src.config import RAW_PRICES, TICKER, START, END


def ingest_data(
    ticker: str = TICKER, start_date: str = START, end_date: str = END
) -> None:
    """
    Extracts historical stock data from Yahoo Finance and saves it as a CSV.

    Args:
        ticker (str, optional): The stock symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """
    print(f"\n⬇️  DOWNLOADING DATA FOR {ticker} ({start_date} to {end_date})...")

    try:
        # 1. download from Yahoo Finance
        df = yf.download(
            ticker, start=start_date, end=end_date, auto_adjust=True, progress=False
        )

        # 2. data validation
        if df.empty:
            raise RuntimeError(f"❌ No data found for {ticker}. Check the symbol.")

        # 3. clean multi-index
        # sometimes YF returns columns like ('Close', 'AAPL') -> want just 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 4. save to raw data folder
        RAW_PRICES.mkdir(parents=True, exist_ok=True)

        out_path = RAW_PRICES / f"{ticker}.csv"

        df.index.name = "Date"
        df.to_csv(out_path)

        print(f"✅ SUCCESS: Saved {len(df)} rows to {out_path}")

    except Exception as e:
        print(f"❌ ERROR in Data Ingestion: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download raw stock data.")

    parser.add_argument(
        "--ticker", type=str, default=TICKER, help=f"Stock symbol (default: {TICKER})"
    )

    parser.add_argument(
        "--start",
        type=str,
        default=START,
        help=f"Start date YYYY-MM-DD (default: {START})",
    )

    parser.add_argument(
        "--end", type=str, default=END, help=f"End date YYYY-MM-DD (default: {END})"
    )

    args = parser.parse_args()

    ingest_data(ticker=args.ticker, start_date=args.start, end_date=args.end)
