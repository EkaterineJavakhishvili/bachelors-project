import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import MODELS_DIR, FIG_DIR
from src.utils import load_data


def evaluate_ensemble(ticker: str = "AAPL"):
    """
    Evaluates a weighted Ensemble Model.

    Methodology:
    - Combines predictions from multiple distinct models (Linear, RF, GB)
    to reduce variance and improve generalizability.

    Weights:
    - 30% Linear Regression (Baseline Trend)
    - 50% Random Forest (Non-linear Patterns)
    - 20% Gradient Boosting (Error Correction)

    Args:
        ticker (str): Stock symbol to train on (Defaults to "AAPL").
    """
    print(f"\n🤝 EVALUATING ENSEMBLE MODEL FOR {ticker}...")

    try:
        # 1. load data
        df = load_data("features", f"{ticker}_features.parquet")

        TARGET = "target_return"
        if TARGET not in df.columns:
            raise ValueError("Target missing. Run features pipeline first.")

        # 2. test split (unseen test data)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()

        feature_cols = [
            c
            for c in df.columns
            if c not in [TARGET, "close", "date", "Date", "actual_price", "pred_price"]
        ]

        print("🧹 Cleaning test data for ensemble...")
        test_df[feature_cols] = test_df[feature_cols].astype(np.float32)
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        test_df = test_df.dropna(subset=feature_cols)

        if test_df.empty:
            raise ValueError("Test set is empty after cleaning!")

        # 3. load all three models
        try:
            lr_model = joblib.load(MODELS_DIR / "linear_regression.pkl")
            rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
            gb_model = joblib.load(MODELS_DIR / "gradient_boosting.pkl")
            print("✅ Successfully loaded LR, RF, and GB models.")
        except FileNotFoundError as e:
            print(f"❌ Critical Error: Could not load models. {e}")
            print("👉 Hint: Run train_linear.py, train_rf.py, and train_gb.py first.")
            return

        # 4. generate individual predictions
        print("⚡ Generating component predictions...")
        pred_lr = lr_model.predict(test_df[feature_cols])
        pred_rf = rf_model.predict(test_df[feature_cols])
        pred_gb = gb_model.predict(test_df[feature_cols])

        # 5. create ensemble
        pred_ensemble = (0.3 * pred_lr) + (0.5 * pred_rf) + (0.2 * pred_gb)

        # 6. reconstruct prices
        price_today = test_df["close"]

        actual_price = price_today * (1 + test_df[TARGET])
        price_lr = price_today * (1 + pred_lr)
        price_rf = price_today * (1 + pred_rf)
        price_gb = price_today * (1 + pred_gb)
        price_ens = price_today * (1 + pred_ensemble)

        # 7. leaderboard
        models_dict = {
            "Linear Reg": price_lr,
            "Random Frst": price_rf,
            "Grad Boost": price_gb,
            ">> ENSEMBLE": price_ens,
        }

        print(f"\n{'Model':<15} | {'RMSE ($)':<10} | {'MAE ($)':<10} | {'R²':<8}")
        print("-" * 52)

        best_rmse = float("inf")
        best_model_name = ""

        for name, pred_prices in models_dict.items():
            rmse = mean_squared_error(actual_price, pred_prices) ** 0.5
            mae = mean_absolute_error(actual_price, pred_prices)
            r2 = r2_score(actual_price, pred_prices)

            print(f"{name:<15} | {rmse:.2f}       | {mae:.2f}       | {r2:.4f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name

        print("-" * 52)
        print(f"🏆 CHAMPION: {best_model_name.strip()} (RMSE: ${best_rmse:.2f})")

        # 8. visualization
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIG_DIR / f"ensemble_results_{ticker}.png"

        plt.figure(figsize=(14, 7))
        zoom = 100  # Last 100 days
        subset_dates = test_df.index[-zoom:]

        # plot actual
        plt.plot(
            subset_dates,
            actual_price[-zoom:],
            label="Actual Price",
            color="black",
            linewidth=2.5,
            alpha=0.8,
        )

        # plot ensemble
        plt.plot(
            subset_dates,
            price_ens[-zoom:],
            label="Ensemble Prediction",
            color="#d62728",
            linestyle="--",
            linewidth=2,
        )

        # plot baseline (linear) for comparison
        plt.plot(
            subset_dates,
            price_lr[-zoom:],
            label="Linear Baseline",
            color="#1f77b4",
            linestyle=":",
            alpha=0.6,
        )

        plt.title(f"Ensemble Strategy vs Market: {ticker} (Last {zoom} Days)")
        plt.ylabel("Stock Price ($)")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(out_path, dpi=150)
        print(f"\n🖼️  Leaderboard Plot saved to: {out_path}")

    except Exception as e:
        print(f"❌ Error in Ensemble Evaluation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ensemble Model.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")

    args = parser.parse_args()
    evaluate_ensemble(ticker=args.ticker)
