"""
Evaluation and plotting for price models
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, FIG_DIR, TICKER, INT_PRICES


def ensure_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR.parent / "metrics.csv").touch(exist_ok=True)


def inverse_trans(target_scaler, arr):
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    return target_scaler.inverse_transform(arr).ravel()


def appends_metrics_csv(
    model_name: str, ticker: str, mae: float, rmse: float, r2: float
):
    metrics_path = FIG_DIR.parent / "metrics.csv"
    row = pd.DataFrame(
        [{"model": model_name, "ticker": ticker, "MAE": mae, "RMSE": rmse, "R2": r2}]
    )

    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        try:
            cur = pd.read_csv(metrics_path)
            cur = pd.concat([cur, row], axis=0, ignore_index=True)
        except Exception:
            cur = row

    else:
        cur = row

    cur.to_csv(metrics_path, index=False)
    print(f"Metrics updated --> {metrics_path}")


def plot_predictions(dates, y, yhat, title: str, out_png: Path):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y, label="Actual")
    plt.plot(dates, yhat, label="Prediction", linestyle="--", marker="x", markersize=3)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Close")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"Figure --> {out_png}")


# Load RF model + scalers + test split, inverse tranform predictions
# print metrics, plot actual vs. predicted
def evaluate_rf(ticker: str = TICKER):
    model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        print("RF model not found")
        return

    rf = joblib.load(model_path)
    scalers = joblib.load(PROC_PRICES / f"{ticker}_scalers.joblib")
    scaler_y = scalers.get("Y")

    test = pd.read_parquet(PROC_PRICES / f"{ticker}_rf_test.parquet")
    target_col = "next_close"
    Xcols = [c for c in test.columns if c != target_col]

    pred_s = rf.predict(test[Xcols]).reshape(-1, 1)
    y_s = test[[target_col]].values
    y = inverse_trans(scaler_y, y_s)
    yhat = inverse_trans(scaler_y, pred_s)

    dates = pd.to_datetime(test.index)

    mae = mean_absolute_error(y, yhat)
    rmse = mean_squared_error(y, yhat) ** 0.5
    r2 = r2_score(y, yhat)
    print(f"[RF test] MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    plot_predictions(
        dates,
        y,
        yhat,
        f"RF Prediction vs. Actual ({ticker})",
        FIG_DIR / "rf_predictions.png",
    )

    appends_metrics_csv("RF", ticker, mae, rmse, r2)


if __name__ == "__main__":
    ensure_dir()
    evaluate_rf()
