import pandas as pd
from src.config import INT_PRICES, PROC_PRICES, TICKER, LOOKBACK, TEST_RATIO, VAL_RATIO


# Build training dataset for the models
def build_features(ticker: str = TICKER):
    # create simple lag features and target to nect-day close prediction

    # load data
    df = pd.read_parquet(INT_PRICES / f"{ticker}.parquet").copy()

    # --- Dataset for Random Forest ---

    # create lag features (previous days' prices & volume)
    LAGS = [1, 2, 3, 5, 10]
    for lag in LAGS:
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag{lag}"] = df["Volume"].shift(lag)

    # trend / volatility context
    df["roll5_mean"] = df["Close"].rolling(5).mean().shift(1)
    df["roll10_mean"] = df["Close"].rolling(10).mean().shift(1)
    df["roll5_std"] = df["Close"].rolling(5).std().shift(1)

    # momentum (recent price changes)
    df["diff1"] = df["Close"].diff(1)  # Close_t - Close_{t-1}
    df["diff3"] = df["Close"].diff(3)  # Close_t - Close_{t-3}

    df["next_close"] = df["Close"].shift(
        -1
    )  # target column, for each day target will be tomorrow's close
    df = (
        df.dropna().copy()
    )  # remove missing rows (ensure every row has complete feature)

    # Split data
    split_idx = int(len(df) * (1 - TEST_RATIO))
    val_split_idx = int(split_idx * (1 - VAL_RATIO))

    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:split_idx]
    test_df = df.iloc[split_idx:]

    # save datasets
    PROC_PRICES.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROC_PRICES / f"{ticker}_rf_train.parquet")
    val_df.to_parquet(PROC_PRICES / f"{ticker}_rf_val.parquet")
    test_df.to_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")

    print(f"Features built for {ticker}")
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
    print("  Columns:", ", ".join(c for c in train_df.columns[:12]), "...")


if __name__ == "__main__":
    build_features()
