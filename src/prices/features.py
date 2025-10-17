import pandas as pd
import numpy as np
from ta import momentum, trend, volatility  # Compute RSI, MACD, ATR, Bollinger
from sklearn.preprocessing import StandardScaler  # Normalization
from joblib import dump
from src.config import INT_PRICES, PROC_PRICES, TICKER, LOOKBACK, TEST_RATIO, VAL_RATIO


# Build training dataset for the models (RF, LSTM)
def build_dataset(ticker: str = TICKER, lookback: int = LOOKBACK):
    df = pd.read_parquet(INT_PRICES / f"{ticker}.parquet")

    # Create additional useful features from base data
    df["ret1"] = df["Close"].pct_change()  # Daily return
    df["rsi14"] = momentum.RSIIndicator(
        df["Close"], window=14
    ).rsi()  # How fast prices are moving
    macd = trend.MACD(df["Close"])
    df["macd"] = macd.macd()  # Diff between two moving averages
    bb = volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["bb_pct"] = (df["Close"] - bb.bollinger_mavg()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )  # Whether price is high or low relative to recent volatility
    df["atr14"] = volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], window=14
    ).average_true_range()  # How much price moves each day on average (last 14 days)

    df = df.dropna().copy()

    # --- Dataset for Random Forest ---

    # For each feature create columns that show the values L days ago
    LAGS = [1, 2, 3, 5, 10]
    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ret1",
        "rsi14",
        "macd",
        "bb_pct",
        "atr14",
    ]:
        for L in LAGS:
            df[f"{col}_lag{L}"] = df[col].shift(L)  # Moves data down by L rows

    df["next_close"] = df["Close"].shift(
        -1
    )  # target column, for each day target will be tomorrow's close
    rf_df = (
        df.dropna().copy()
    )  # remove missing rows (ensure every row has complete feature)

    # Split data
    split_idx = int(len(rf_df) * (1 - TEST_RATIO))
    rf_train = rf_df.iloc[:split_idx].copy()
    rf_test = rf_df.iloc[split_idx:].copy()

    # Validation portion (model will be tuned/checked on validation data before testing)
    val_idx = int(len(rf_train) * (1 - VAL_RATIO))
    rf_tr, rf_val = rf_train.iloc[:val_idx].copy(), rf_train.iloc[val_idx:].copy()

    # Data Normalization
    rf_features = [c for c in rf_tr.columns if c not in ["next_close"]]
    scaler_X = StandardScaler().fit(rf_tr[rf_features])
    scaler_Y = StandardScaler().fit(rf_tr[["next_close"]])

    for part in [rf_tr, rf_val, rf_test]:
        part[rf_features] = part[rf_features].astype("float64").copy()
        part.loc[:, rf_features] = scaler_X.transform(part[rf_features])
        part.loc[:, "next_close"] = scaler_Y.transform(part[["next_close"]])

    # Save RF's dataset
    rf_tr.to_parquet(PROC_PRICES / f"{ticker}_rf_train.parquet")
    rf_val.to_parquet(PROC_PRICES / f"{ticker}_rf_val.parquet")
    rf_test.to_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")
    dump({"X": scaler_X, "Y": scaler_Y}, PROC_PRICES / f"{ticker}_scalers.joblib")

    # --- Dataset for LSTM ---

    # feature set for LSTM sequential data
    seq_data = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "ret1",
        "rsi14",
        "macd",
        "bb_pct",
        "atr14",
    ]
    seq_df = df.dropna().copy()

    # Split data
    train_seq = seq_df.iloc[:split_idx].copy()
    te_seq = seq_df.iloc[split_idx:].copy()
    val_cut = int(len(train_seq) * (1 - VAL_RATIO))
    tr_seq, va_seq = train_seq.iloc[:val_cut].copy(), train_seq.iloc[val_cut:].copy()

    # Data Normalization
    scaler_X2 = StandardScaler().fit(tr_seq[seq_data])
    scaler_Y2 = StandardScaler().fit(
        tr_seq[["Close"]].shift(-1).dropna()
    )  # fit on next day close of train portion

    # Build 3D tensor window for LSTM (samples, lookback, features)
    # Each sample's target is next day after window ends
    def build_windows(frame):
        Xs, Ys, idx = [], [], []
        arrX = scaler_X2.transform(frame[seq_data])
        arrY = scaler_Y2.transform(frame[["Close"]])  # t + 1
        for i in range(lookback, len(frame) - 1):
            Xs.append(arrX[i - lookback : i, :])  # t - lookback..t
            Ys.append(arrY[i + 1, 0])  # predict close at t + 1
            idx.append(frame.index[i])
        return np.array(Xs), np.array(Ys), idx

    # Create arrays for each portion of data
    X_tr, Y_tr, _ = build_windows(tr_seq)
    X_va, Y_va, _ = build_windows(va_seq)
    X_te, Y_te, te_idx = build_windows(te_seq)  # te_idx have the dates for plotting

    # Save LSTM's dataset
    np.save(PROC_PRICES / f"{ticker}_lstm_X_tr.npy", X_tr)
    np.save(PROC_PRICES / f"{ticker}_lstm_Y_tr.npy", Y_tr)
    np.save(PROC_PRICES / f"{ticker}_lstm_X_va.npy", X_va)
    np.save(PROC_PRICES / f"{ticker}_lstm_Y_va.npy", Y_va)
    np.save(PROC_PRICES / f"{ticker}_lstm_X_te.npy", X_te)
    np.save(PROC_PRICES / f"{ticker}_lstm_Y_te.npy", Y_te)
    pd.Series(te_idx).to_pickle(PROC_PRICES / f"{ticker}_lstm_test_index.pkl")
    dump(
        {"X": scaler_X2, "Y": scaler_Y2}, PROC_PRICES / f"{ticker}_lstm_scalers.joblib"
    )

    print("RF and LSTM datasets were built")


if __name__ == "__main__":
    build_dataset()
