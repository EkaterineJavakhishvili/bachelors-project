import argparse
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import DATA_DIR, FIG_DIR, MODELS_DIR
from src.utils import load_data


def run_evaluation(ticker: str, model_filename: str):
    """
    Universal Evaluator: loads ANY trained model (.pkl) and tests it.

    Args:
        ticker (str): The stock symbol (e.g., 'AAPL').
        model_filename (str): The filename saved by train sctips (e.g., 'random_forest.pkl').
    """
    print(f"\n📊 EVALUATING: {model_filename} on {ticker}...")

    try:
        # 1. load data
        df = load_data("features", f"{ticker}_features.parquet")

        # split (using test set only)
        test_size = int(len(df) * 0.2)
        test_df = df.iloc[-test_size:].copy()

        feature_cols = [
            c for c in df.columns if c not in ["target_return", "close", "date", "Date"]
        ]

        # 3. load the model using config path
        model_path = MODELS_DIR / model_filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found: {model_path}\n"
                f"👉 Hint: Check if the file exists in 'models/price_agent/' or run training first."
            )

        print(f"🤖 Loading model from: {model_path.name}")
        model = joblib.load(model_path)

        # 4. predict & reconsruct prices
        test_df["pred_return"] = model.predict(test_df[feature_cols])

        # price tomorrow = price today * (1 + predicted return)
        test_df["actual_price"] = test_df["close"] * (1 + test_df["target_return"])
        test_df["pred_price"] = test_df["close"] * (1 + test_df["pred_return"])

        # 5. metrics
        y_true = test_df["actual_price"]
        y_pred = test_df["pred_price"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)

        print("\n" + "-" * 40)
        print(f"🏆 PERFORMANCE REPORT: {model_filename}")
        print("-" * 40)
        print(f"✅ MAE (Avg Error): ${mae:.2f}")
        print(f"✅ RMSE          : ${rmse:.2f}")
        print(f"✅ R² Score      : {r2:.4f}")
        print("-" * 40)

        # 6. plotting
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        img_name = f"eval_{ticker}_{model_filename.replace('.pkl', '')}.png"
        save_path = FIG_DIR / img_name

        plt.figure(figsize=(12, 6))
        subset = test_df.iloc[-100:]  # last 100 days

        plt.plot(
            subset.index,
            subset["actual_price"],
            label="Actual Price",
            linewidth=2,
            color="black",
            alpha=0.7,
        )
        plt.plot(
            subset.index,
            subset["pred_price"],
            label="Model Prediction",
            linestyle="--",
            linewidth=2,
            color="orange",
        )

        plt.title(f"Model Evaluation: {model_filename} ({ticker})")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(save_path)
        print(f"🖼️  Chart saved to: {save_path}")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a specific trained model.")
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker (default: AAPL)"
    )
    parser.add_argument(
        "--model", type=str, default="random_forest.pkl", help="Model filename to load"
    )

    args = parser.parse_args()

    run_evaluation(ticker=args.ticker, model_filename=args.model)
