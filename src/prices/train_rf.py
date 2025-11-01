"""
Train RandomForest baseline model to predict next-day close price.
"""

import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, TICKER


# Convert (n,1) or(n,) array of scaled y back to real price units
def inverse_transform_target(y_scaled, target_scaler):
    if y_scaled.ndim == 1:
        y_scaled = y_scaled.reshape(-1, 1)

    return target_scaler.inverse_transform(y_scaled).ravel()


# Loads processed parquet files created by features.py, trains RandomForestRegressor,
# evaluates on val and test, inverse-transforms predictions
# to price units, prints metrics, and saves model
def train_rf(ticker: str = TICKER):
    # Load splits and scalers
    trn = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_train.parquet")
    val = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_val.parquet")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")
    scalers = joblib.load(PROC_PRICES / f"{ticker}_scalers.joblib")
    scaler_y = scalers.get("Y")

    # features (without target)
    target_col = "next_close"
    Xcols = [c for c in trn.columns if c != target_col]

    # Model definition
    rf = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
    )

    # Train
    rf.fit(trn[Xcols], trn[target_col])

    def evaluate(split_name: str, df: pd.DataFrame):
        pred_scaled = rf.predict(df[Xcols]).reshape(-1, 1)
        y_scaled = df[[target_col]].values

        # Inverse to original $ price
        y = inverse_transform_target(y_scaled, scaler_y)
        yhat = inverse_transform_target(pred_scaled, scaler_y)

        mae = mean_absolute_error(y, yhat)
        rmse = mean_squared_error(y, yhat) ** 0.5
        r2 = r2_score(y, yhat)
        print(f"[RF {split_name}] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

        return mae, rmse, r2

    evaluate("val", val)
    evaluate("test", test)

    # Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / "random_forest.pkl"
    joblib.dump(rf, out_path)
    print(f"Saved RF model --> {out_path}")


if __name__ == "__main__":
    train_rf()
