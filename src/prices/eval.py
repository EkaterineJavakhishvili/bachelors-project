"""
Evaluation and plotting for price models
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.config import PROC_PRICES, MODELS_DIR, FIG_DIR, TICKER, INT_PRICES


TARGET = "next_close"


def evaluate_and_plot(ticker: str = TICKER):
    # load model and test split 
    # (!!! CHANGE DEPENDING ON WHICH MODEL YOU'RE USING !!!)
    rf = joblib.load(MODELS_DIR / "linear_regression.pkl")
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")

    # guard against NaNs
    Xcols = [c for c in test.columns if c != TARGET]
    test = test.dropna(subset=Xcols + [TARGET]).copy()

    # predict
    y_true = test[TARGET].values
    y_pred = rf.predict(test[Xcols])
    dates = pd.to_datetime(test.index)

    # plot
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "rf_predictions_test.png"
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(
        dates, y_pred, label="Prediction", linestyle="--", marker="x", markersize=2
    )
    plt.title(f"RF Predictions vs Actual ({ticker}) • TEST")
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    print(f"Figure → {out}")


if __name__ == "__main__":
    evaluate_and_plot()
