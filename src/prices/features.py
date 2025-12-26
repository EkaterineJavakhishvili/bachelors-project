import pandas as pd
import numpy as np
import pandas_ta as ta
from src.config import INT_PRICES, PROC_PRICES, TICKER, TEST_RATIO, VAL_RATIO


def build_features(ticker: str = TICKER):
    """
    Builds features using pandas_ta with robust cleaning for Infinite values.
    """
    # 1. Load Data
    df = pd.read_parquet(INT_PRICES / f"{ticker}.parquet").copy()

    # --- 2. BASE CALCULATIONS ---
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    # --- 3. TECHNICAL INDICATORS (via pandas_ta) ---
    # RSI (14)
    df.ta.rsi(length=14, append=True)

    # MACD (12, 26, 9)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # ATR (14)
    df.ta.atr(length=14, append=True)

    # --- 4. NORMALIZATION & CLEANUP ---
    # Normalize RSI (0-100 -> 0-1)
    df["rsi_norm"] = df["RSI_14"] / 100.0

    # Normalize ATR (Relative to price)
    df["atr_rel"] = df["ATRr_14"] / df["Close"]

    # MACD Histogram
    df["macd_hist"] = df["MACDh_12_26_9"]

    # Rolling Volatility
    df["volatility_20"] = df["log_return"].rolling(20).std()

    # Lags
    LAGS = [1, 2, 3, 5]
    for lag in LAGS:
        df[f"return_lag{lag}"] = df["log_return"].shift(lag)

    # --- 5. TARGET DEFINITION ---
    df["target_return"] = df["log_return"].shift(-1)
    df["price_today"] = df["Close"]

    # --- !!! CRITICAL FIX: SANITIZE DATA !!! ---
    # 1. Replace Infinity with NaN (Fixes the Linear Regression crash)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 2. Drop NaNs (drops the rows with Inf, plus the startup rows for MACD/RSI)
    df = df.dropna().copy()

    # --- 6. SPLIT DATA ---
    split_idx = int(len(df) * (1 - TEST_RATIO))
    val_split_idx = int(split_idx * (1 - VAL_RATIO))

    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:split_idx]
    test_df = df.iloc[split_idx:]

    # --- 7. SELECT FINAL COLUMNS ---
    feature_cols = [
        "return_lag1",
        "return_lag2",
        "return_lag3",
        "return_lag5",
        "rsi_norm",
        "macd_hist",
        "atr_rel",
        "volatility_20",
    ]
    final_cols = feature_cols + ["target_return", "price_today"]

    PROC_PRICES.mkdir(parents=True, exist_ok=True)
    train_df[final_cols].to_parquet(PROC_PRICES / f"{ticker}_train.parquet")
    val_df[final_cols].to_parquet(PROC_PRICES / f"{ticker}_val.parquet")
    test_df[final_cols].to_parquet(PROC_PRICES / f"{ticker}_test.parquet")

    print(f"Features built for {ticker} (Infinities removed!)")
    print(f"Train Shape: {train_df.shape}")


if __name__ == "__main__":
    build_features()
