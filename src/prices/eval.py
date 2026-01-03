"""
Evaluation and plotting for price models (Reconstructed from Returns)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, FIG_DIR, TICKER

# !!! CHANGED TARGET !!!
TARGET = "target_return"


def evaluate_and_plot(ticker: str = TICKER):
    # Load model (Swap this for 'gradient_boosting.pkl' or 'linear_regression.pkl' to test others)
    model_name = "random_forest.pkl"
    model = joblib.load(MODELS_DIR / model_name)

    test = pd.read_parquet(PROC_PRICES / f"{ticker}_test.parquet")

    # Features: Exclude target and helper column
    Xcols = [c for c in test.columns if c not in [TARGET, "price_today"]]
    test = test.dropna(subset=Xcols + [TARGET]).copy()

    # 1. Predict Returns
    # The model outputs: "I think price will change by +0.01 (1%)"
    pred_log_returns = model.predict(test[Xcols])

    # 2. Reconstruct Prices
    # Formula: Price_Tomorrow = Price_Today * exp(Log_Return)
    # We use the 'price_today' column we saved in features.py
    test["pred_price"] = test["price_today"] * np.exp(pred_log_returns)
    test["actual_price"] = test["price_today"] * np.exp(test[TARGET])

    # 3. Calculate Metrics on PRICES (Real world accuracy)
    y_true = test["actual_price"]
    y_pred = test["pred_price"]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    print(f"--- RECONSTRUCTED PRICE METRICS ({model_name}) ---")
    print(f"[TEST] MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")

    # 4. Plot
    dates = pd.to_datetime(test.index)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"pred_{model_name.replace('.pkl', '')}.png"

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label="Actual Price", linewidth=2)
    plt.plot(dates, y_pred, label="Predicted Price", linestyle="--", color="orange")

    plt.title(
        f"Prediction vs Actual ({ticker}) • Reconstructed from Returns • {model_name}"
    )
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Figure saved -> {out}")


if __name__ == "__main__":
    evaluate_and_plot()
