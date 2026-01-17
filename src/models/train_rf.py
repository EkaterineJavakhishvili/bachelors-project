import argparse
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config import MODELS_DIR
from src.utils import load_data


def train_rf(ticker: str = "AAPL"):
    """
    Trains a Random Forest Regressor agent.

    Advantages:
    - non-linear: can capture complex patterns
    - robust: less sensitive to outliers then deep learning

    Args:
        ticker (str): Stock symbol to train on (Defaults to "AAPL").
    """
    print(f"\n🌲 TRAINING RANDOM FOREST AGENT FOR {ticker}...")

    try:
        # 1. load data
        df = load_data("features", f"{ticker}_features.parquet")

        TARGET = "target_return"
        if TARGET not in df.columns:
            raise ValueError(f"Target '{TARGET}' missing. Run features pipeline first.")

        # 2. split (80-20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # define features
        X_cols = [
            c
            for c in df.columns
            if c not in [TARGET, "close", "date", "Date", "actual_price", "pred_price"]
        ]

        print(f"   Data Split: {len(train_df)} Train / {len(test_df)} Test rows")
        print(f"   Features: {len(X_cols)} inputs")

        # 3. initialize model
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

        # 4. train
        print("⚡ Fitting Random Forest...")
        rf.fit(train_df[X_cols], train_df[TARGET])

        # 5. evaluate
        print("\n--- Model Performance (Test Set) ---")
        y_true = test_df[TARGET]
        y_pred = rf.predict(test_df[X_cols])

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)

        print(f"✅ MAE  : {mae:.5f}")
        print(f"✅ RMSE : {rmse:.5f}")
        print(f"✅ R²   : {r2:.5f}")

        # 6. feature importance
        print("\n--- Top 5 Drivers of Price ---")
        importances = rf.feature_importances_
        feat_imp = pd.DataFrame({"Feature": X_cols, "Importance": importances})
        print(
            feat_imp.sort_values("Importance", ascending=False)
            .head(5)
            .to_string(index=False)
        )

        # 7. save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MODELS_DIR / "random_forest.pkl"

        joblib.dump(rf, out_path)
        print(f"\n💾 Model saved to: {out_path}")

    except Exception as e:
        print(f"❌ Error during training: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")

    args = parser.parse_args()
    train_rf(ticker=args.ticker)
