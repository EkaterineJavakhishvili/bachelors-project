import pandas as pd
import numpy as np
import pandas_ta as ta
from src.config import DATA_DIR


def build_features(ticker="AAPL"):
    print(f"Building features for {ticker}...")

    # 1. Load Price Data
    price_path = DATA_DIR / "raw" / "prices" / f"{ticker}.csv"
    if not price_path.exists():
        print(f"❌ Price data not found: {price_path}")
        return

    df = pd.read_csv(price_path)

    # --- FIX: Standardize Column Names (Handle date vs Date, close vs Close) ---
    # This ensures the script works even if your CSV has lowercase names
    df = df.rename(
        columns={
            "date": "Date",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "open": "Open",
            "volume": "Volume",
        }
    )

    # Ensure Date is actually a column (if it was the index, reset it)
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})

    # ---------------------------------------------------------------------------

    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.date
    df = df.set_index("Date").sort_index()

    # 2. Calculate Returns (Input Features)
    df["return_lag1"] = df["Close"].pct_change()
    df["return_lag2"] = df["Close"].pct_change(2)
    df["return_lag3"] = df["Close"].pct_change(3)
    df["return_lag5"] = df["Close"].pct_change(5)

    # 3. Technical Indicators (RSI, MACD, Volatility)
    df["rsi"] = ta.rsi(df["Close"], length=14)
    df["rsi_norm"] = df["rsi"] / 100.0  # Normalize 0-1

    macd = ta.macd(df["Close"])
    df["macd_hist"] = macd["MACDh_12_26_9"]

    df["atr"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["atr_rel"] = df["atr"] / df["Close"]

    df["volatility_20"] = df["return_lag1"].rolling(20).std()

    # 4. Create TARGET (What we want to predict)
    # We want to predict tomorrow's return today.
    # So we shift returns BACKWARDS by 1 day.
    df["target_return"] = df["return_lag1"].shift(-1)

    # Keep the Close price for backtesting simulations later
    df["close"] = df["Close"]

    # 5. Merge Sentiment (If available)
    sent_path = DATA_DIR / "processed" / "news" / f"{ticker}_sentiment.parquet"
    if sent_path.exists():
        print(f"✅ Found Sentiment Data! Merging {sent_path.name}...")
        sent_df = pd.read_parquet(sent_path)
        # sent_df index is likely datetime, convert to date if needed
        if isinstance(sent_df.index, pd.DatetimeIndex):
            sent_df.index = sent_df.index.date

        df = df.join(sent_df, how="left")

        # Fill missing sentiment with 0 (neutral) or forward fill
        df["sentiment_finbert"] = df["sentiment_finbert"].fillna(0)

        # Sentiment lags
        df["sent_lag1"] = df["sentiment_finbert"].shift(1)
        df["sent_lag2"] = df["sentiment_finbert"].shift(2)
        df["sent_roll5"] = df["sentiment_finbert"].rolling(5).mean()
    else:
        print("⚠️ No sentiment data found. Skipping sentiment features.")

    # 6. Drop NaNs (created by lags/rolling windows)
    df = df.dropna()

    # 7. Define Columns to Save
    feature_cols = [
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "return_lag5",
        "rsi_norm",
        "macd_hist",
        "atr_rel",
        "volatility_20",
        "sent_lag1",
        "sent_lag2",
        "sent_roll5",
        "target_return",
        "close",
    ]

    # Filter only existing columns
    final_cols = [c for c in feature_cols if c in df.columns]

    out_path = DATA_DIR / "processed" / "features" / f"{ticker}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df[final_cols].to_parquet(out_path)
    print(f"Features saved. Columns: {final_cols}")


if __name__ == "__main__":
    build_features()
