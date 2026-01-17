import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import MODELS_DIR
from src.utils import load_data


def train_linear(ticker: str = "AAPL"):
    """
    Trains a Ridge Regression model with Rebust Scaling.

    Why this model?
    - RobustScaler: Handles financial outliers (e.g., market crashes) better than Standard scaling.
    - Ridge Regression: Uses L2 regularization to handle multicollinearity between technical indicators.

    Args:
        ticker (str, optional): Stock symbol to train on (Defaults to "AAPL").
    """
    print(f"\n🧠 TRAINING LINEAR AGENT FOR {ticker}...")

    try:
        # 1. load data
        df = load_data("features", f"{ticker}_features.parquet")

        TARGET = "target_return"
        if TARGET not in df.columns:
            raise ValueError(f"Target '{TARGET}' missing. Run features pipeline first.")

        # 2. time-series split (80% train, 20% test)
        # splitting chronologically to respect time (no looking into future)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # Define Features (exclude target and helper cols)
        X_cols = [
            c
            for c in df.columns
            if c not in [TARGET, "close", "date", "Date", "actual_price", "pred_price"]
        ]

        print(f"   Data Split: {len(train_df)} Train / {len(test_df)} Test rows")
        print(f"   Features: {len(X_cols)} inputs (RSI, MACD, Sentiment, etc.)")

        # 3. data cleaning
        for data in [train_df, test_df]:
            data[X_cols] = data[X_cols].astype(np.float32)
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(subset=X_cols + [TARGET], inplace=True)

        if train_df.empty:
            raise RuntimeError("Training data is empty after cleaning!")

        # 4. create pipeline
        model = make_pipeline(RobustScaler(), Ridge(alpha=1.0, random_state=42))

        # 5. train
        print("⚡ Fitting Ridge Regression...")
        model.fit(train_df[X_cols], train_df[TARGET])

        # 6. evaluate
        print("\n--- Model Performance (Test Set) ---")
        y_true = test_df[TARGET]
        y_pred = model.predict(test_df[X_cols])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)

        print(f"✅ MAE  : {mae:.5f}")
        print(f"✅ RMSE : {rmse:.5f}")
        print(f"✅ R²   : {r2:.5f}")

        # 7. analyze coefficients
        print("\n--- Feature Importance (Top 5) ---")
        coefs = model.named_steps["ridge"].coef_
        feat_imp = pd.DataFrame({"Feature": X_cols, "Coef": coefs})
        feat_imp["AbsCoef"] = feat_imp["Coef"].abs()
        print(
            feat_imp.sort_values("AbsCoef", ascending=False)
            .head(5)
            .to_string(index=False)
        )

        # 8. save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / "linear_regression.pkl"

        joblib.dump(model, out_path)
        print(f"\n💾 Model saved to: {out_path}")

    except Exception as e:
        print(f"❌ Error during training: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Linear Regression (Ridge) model."
    )
    parser.add_argument(
        "--ticker", type=str, default="AAPL", help="Stock ticker to train on"
    )

    args = parser.parse_args()
    train_linear(ticker=args.ticker)
