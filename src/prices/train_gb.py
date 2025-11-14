import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, TICKER
import joblib


def train_gb(ticker: str = TICKER):
    train = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_train.parquet")
    val = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_val.parquet")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")

    TARGET = "next_close"
    Xcols = [c for c in train.columns if c != TARGET]

    gb = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42
    )
    gb.fit(train[Xcols], train[TARGET])

    # evaluate
    for name, df in [("VAL", val), ("TEST", test)]:
        df = df.dropna(subset=Xcols + [TARGET])
        y_true = df[TARGET]
        y_pred = gb.predict(df[Xcols])
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"[{name}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "gradient_boosting.pkl"
    joblib.dump(gb, out)
    print(f"Saved model → {out}")


if __name__ == "__main__":
    train_gb()
