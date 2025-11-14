"""
Train RandomForest baseline model to predict next-day close price.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, TICKER


def train_rf(ticker: str = TICKER):
    # load processed data from features.py
    train = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_train.parquet")
    val = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_val.parquet")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")

    TARGET = "next_close"
    Xcols = [c for c in train.columns if c != TARGET]

    # train
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
    rf.fit(train[Xcols], train[TARGET])

    # evaluate
    for name, df in [("val", val), ("test", test)]:
        # drop rows with NaN
        df_e = df.dropna(subset=Xcols + [TARGET]).copy()
        
        # if any rows where dropped
        dropped = len(df) - len(df_e)
        if dropped:
            print(f"[{name}] dropped {dropped} rows with NaNs before evaluation")
        
        
        y_true = df_e[TARGET].astype(float).values
        y_pred = rf.predict(df_e[Xcols].astype(float))
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"[{name.upper()}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "random_forest.pkl"
    joblib.dump(rf, out)
    print(f"saved model --> {out}")


if __name__ == "__main__":
    train_rf()
