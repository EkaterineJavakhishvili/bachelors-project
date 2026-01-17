import argparse
import pandas as pd
import pandas_ta as ta
from src.config import DATA_DIR
from src.utils import load_data, standardize_columns


def build_features(ticker: str = "AAPL") -> None:
    """
    Main pipeline to generate technical and sentiment features.

    Process:
    1. Extract: load raw prices
    2. Transform: calculated RSI, MACD, and merge Sentiment
    3. Load: save to Parquet

    Args:
        ticker (str): Stock symbol to process (Defaults to "AAPL").
    """
    print(f"\n🏗️  STARTING FEATURE ENGINEERING FOR {ticker}...")

    try:
        # 1. load price data
        # uses the utility function for rebust path handling
        df = load_data("prices", f"{ticker}.parquet")

        if "Date" not in df.columns and df.index.name == "Date":
            df = df.reset_index()

        df = standardize_columns(df)

        # convert data to ensure datetime index
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
        df = df.set_index("Date").sort_index()

        # 2. technical indicators (vectorized for performance)
        print("📊 Calculating technical indicators...")
        df["return_lag1"] = df["Close"].pct_change().shift(1)
        df["return_lag2"] = df["Close"].pct_change().shift(2)
        df["return_lag3"] = df["Close"].pct_change().shift(3)
        df["return_lag5"] = df["Close"].pct_change().shift(5)

        # RSI (relative strength index) - normalized to 0-1
        df["rsi"] = ta.rsi(df["Close"], length=14) / 100.0

        # MACD (moving average convergence divergence)
        df["macd_hist"] = ta.macd(df["Close"])["MACDh_12_26_9"]

        # volatility (rolling standard deviation)
        df["volatility_20"] = df["return_lag1"].rolling(20).std()

        # 3. create target
        # shifting returns backward so today's features predict tomorrow's return
        df["target_return"] = df["Close"].pct_change().shift(-1)
        df["close"] = df["Close"]  # keep raw close price for backtesting visualization

        # 4. merge sentiment
        try:
            print("🧠 Merging FinBERT sentiment data...")
            sent_df = load_data("news", f"{ticker}_sentiment.parquet")

            # align index types
            if isinstance(sent_df.index, pd.DatetimeIndex):
                sent_df.index = sent_df.index.date

            # left join ensures keeping all price days, even if no news exists
            df = df.join(sent_df, how="left")

            # fill missing sentiment days with 0 (neutral)
            df["sentiment_finbert"] = df["sentiment_finbert"].fillna(0)

            # lag sentiment to prevent data leakage (use yesterday's news for today's trade)
            df["sent_lag1"] = df["sentiment_finbert"].shift(1)

        except FileNotFoundError:
            print("⚠️ Warning: No sentiment data found. Skipping NLP features.")

        # 5. save data
        df = df.dropna()

        # select only the columns needed for the model
        cols = [
            "return_lag1",
            "return_lag2",
            "return_lag3",
            "return_lag5",
            "rsi",
            "macd_hist",
            "volatility_20",
            "sent_lag1",
            "target_return",
            "close",
        ]
        final_cols = [c for c in cols if c in df.columns]

        out_path = DATA_DIR / "processed" / "features" / f"{ticker}_features.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        df[final_cols].to_parquet(out_path)

        print(f"✅ SUCCESS: Features saved to {out_path.name}")

    except Exception as e:
        print(f"❌ ERROR in Feature Engineering: {e}")


if __name__ == "__main__":
    # allow different scenarios via CLI arguments
    parser = argparse.ArgumentParser(
        description="Generate features for a specific stock."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="The stock symbol to process (default: AAPL)",
    )

    args = parser.parse_args()
    build_features(ticker=args.ticker)
