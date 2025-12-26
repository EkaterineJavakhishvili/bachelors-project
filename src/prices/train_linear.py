import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, TICKER
import joblib


def train_linear(ticker: str = TICKER):
    """
    Train Ridge Regression using RobustScaler.
    RobustScaler handles financial outliers better than StandardScaler,
    preventing overflow warnings.
    """

    # 1. Load Data
    train = pd.read_parquet(PROC_PRICES / f"{ticker}_train.parquet")
    val = pd.read_parquet(PROC_PRICES / f"{ticker}_val.parquet")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_test.parquet")

    TARGET = "target_return"
    Xcols = [c for c in train.columns if c not in [TARGET, "price_today"]]

    # 2. Aggressive Data Cleaning
    # Convert to float32 (fixes some specific macOS/numpy overflow bugs)
    for df in [train, val, test]:
        df[Xcols] = df[Xcols].astype(np.float32)

    # Drop Infinite/NaN values
    train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=Xcols + [TARGET])
    val = val.replace([np.inf, -np.inf], np.nan).dropna(subset=Xcols + [TARGET])
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=Xcols + [TARGET])

    print(f"--- Training Ridge Regression ({len(Xcols)} features) ---")

    # 3. Create Pipeline with RobustScaler
    model = make_pipeline(RobustScaler(), Ridge(alpha=1.0, random_state=42))

    # 4. Train
    model.fit(train[Xcols], train[TARGET])

    # 5. Check Coefficients
    coefs = model.named_steps["ridge"].coef_
    print("Feature Coefficients:")
    for feature, coef in zip(Xcols, coefs):
        print(f"  {feature:<15}: {coef:.5f}")

    # 6. Evaluate
    print(f"\n--- Model Performance (Returns) ---")
    for name, df in [("VAL", val), ("TEST", test)]:
        y_true = df[TARGET]
        y_pred = model.predict(df[Xcols])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        print(f"[{name}]  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # 7. Save
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = MODELS_DIR / "linear_regression.pkl"
    joblib.dump(model, out)
    print(f"[OK] Saved Robust Ridge model → {out}")


if __name__ == "__main__":
    train_linear()
