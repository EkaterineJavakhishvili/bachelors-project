import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import PROC_PRICES, MODELS_DIR, FIG_DIR, TICKER

TARGET = "target_return"


def evaluate_ensemble(ticker: str = TICKER):
    print(f"\n--- ENSEMBLE EVALUATION ({ticker}) ---")

    # 1. Load Test Data
    test = pd.read_parquet(PROC_PRICES / f"{ticker}_test.parquet")
    Xcols = [c for c in test.columns if c not in [TARGET, "price_today"]]

    # 2. MATCHING CLEANING LOGIC (Crucial to stop warnings)
    # We must treat the test data exactly like the training data
    test[Xcols] = test[Xcols].astype(np.float32)
    test = test.replace([np.inf, -np.inf], np.nan).dropna(subset=Xcols + [TARGET])

    # 3. Load Models
    try:
        rf = joblib.load(MODELS_DIR / "random_forest.pkl")
        gb = joblib.load(MODELS_DIR / "gradient_boosting.pkl")
        lr = joblib.load(MODELS_DIR / "linear_regression.pkl")
    except FileNotFoundError as e:
        print(f"❌ Error: Missing model file. {e}")
        return

    # 4. Generate Predictions
    print("Generating predictions...")
    pred_lr = lr.predict(test[Xcols])
    pred_rf = rf.predict(test[Xcols])
    pred_gb = gb.predict(test[Xcols])

    # 5. Create Ensemble (Weighted Average)
    # Weights: 50% Linear (Base), 30% RF (Structure), 20% GB (Correction)
    pred_ensemble = (0.5 * pred_lr) + (0.3 * pred_rf) + (0.2 * pred_gb)

    # 6. Reconstruct Prices
    price_today = test["price_today"]
    actual_price = price_today * np.exp(test[TARGET])

    price_lr = price_today * np.exp(pred_lr)
    price_rf = price_today * np.exp(pred_rf)
    price_gb = price_today * np.exp(pred_gb)
    price_ens = price_today * np.exp(pred_ensemble)

    # 7. Calculate & Compare Metrics
    models_dict = {
        "Linear Reg ": price_lr,
        "Random Frst": price_rf,
        "Grad Boost ": price_gb,
        ">> ENSEMBLE": price_ens,
    }

    print(f"\n{'Model':<15} | {'RMSE':<8} | {'MAE':<8} | {'R2':<8}")
    print("-" * 46)

    best_rmse = float("inf")
    best_model = ""

    for name, pred in models_dict.items():
        rmse = mean_squared_error(actual_price, pred) ** 0.5
        mae = mean_absolute_error(actual_price, pred)
        r2 = r2_score(actual_price, pred)

        print(f"{name:<15} | {rmse:.3f}    | {mae:.3f}    | {r2:.4f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = name

    print("-" * 46)
    print(f"🏆 WINNER: {best_model.strip()} (RMSE: {best_rmse:.3f})")

    # 8. Visualization
    dates = pd.to_datetime(test.index)
    zoom = 100  # Last 100 days

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))

    plt.plot(
        dates[-zoom:],
        actual_price[-zoom:],
        label="Actual",
        color="black",
        linewidth=2,
        alpha=0.7,
    )
    plt.plot(
        dates[-zoom:],
        price_ens[-zoom:],
        label="Ensemble",
        color="red",
        linestyle="--",
        linewidth=2,
    )
    plt.plot(
        dates[-zoom:],
        price_lr[-zoom:],
        label="Linear (Base)",
        color="blue",
        linestyle=":",
        alpha=0.5,
    )

    plt.title(f"Ensemble vs Linear Baseline ({ticker}) • Last {zoom} Days")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path = FIG_DIR / "ensemble_results.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nPlot saved to: {out_path}")


if __name__ == "__main__":
    evaluate_ensemble()
