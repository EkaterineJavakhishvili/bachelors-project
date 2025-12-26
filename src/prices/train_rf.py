import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, TICKER


def train_rf(ticker: str = TICKER):
    train = pd.read_parquet(PROC_PRICES / f"{ticker}_train.parquet")
    val = pd.read_parquet(PROC_PRICES / f"{ticker}_val.parquet")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_test.parquet")

    # !!! CHANGED TARGET !!!
    TARGET = "target_return"

    # Features are everything EXCEPT target and the helper 'price_today'
    Xcols = [c for c in train.columns if c not in [TARGET, "price_today"]]

    rf = RandomForestRegressor(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(train[Xcols], train[TARGET])

    # Evaluate on Returns first (to see if model learns the movement)
    print(f"--- Model Performance on Returns (Not Prices) ---")
    for name, df in [("VAL", val), ("TEST", test)]:
        y_true = df[TARGET]
        y_pred = rf.predict(df[Xcols])

        # We generally look at RMSE for returns
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"[{name}] RMSE={rmse:.5f} R2={r2:.5f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "random_forest.pkl"
    joblib.dump(rf, out)
    print(f"Saved model -> {out}")


if __name__ == "__main__":
    train_rf()
