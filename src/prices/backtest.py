import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.config import DATA_DIR


def run_backtest(ticker="AAPL", initial_capital=10000.0):
    # 1. Load the data
    feature_path = DATA_DIR / "processed" / "features" / f"{ticker}_features.parquet"
    print(f"🔍 Loading data from: {feature_path}")

    if not feature_path.exists():
        print("❌ Data not found. Run 'src.prices.features' first.")
        return

    df = pd.read_parquet(feature_path).dropna()

    # --- DEBUGGING BLOCK ---
    print(f"✅ Loaded {len(df)} rows.")
    print(f"📋 Columns found in file: {df.columns.tolist()}")

    if "target_return" not in df.columns:
        print("❌ CRITICAL ERROR: 'target_return' is MISSING from the dataframe.")
        print(
            "   This means the features file was saved incorrectly or loaded the wrong file."
        )
        return
    # -----------------------

    # 2. Split Data (Keep the last 20% for testing, strictly time-ordered)
    test_size = int(len(df) * 0.2)
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:].copy()

    # 3. Define Features & Target
    # Explicitly excluding non-feature columns
    features = [
        c for c in df.columns if c not in ["target_return", "close", "date", "Date"]
    ]

    print(f"🧠 Training with features: {features}")

    X_train, y_train = train_df[features], train_df["target_return"]
    X_test = test_df[features]

    # 4. Train the Winner (Random Forest)
    print(f"🤖 Training Random Forest on {len(train_df)} days...")
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 5. Generate Predictions
    test_df["pred_return"] = model.predict(X_test)

    # =========================================================
    # 💰 STRATEGY SIMULATION
    # =========================================================

    threshold = 0.001  # 0.1% threshold

    # 1. Signal: 1 = Long, -1 = Short, 0 = Cash
    test_df["position"] = 0
    test_df.loc[test_df["pred_return"] > threshold, "position"] = 1
    test_df.loc[test_df["pred_return"] < -threshold, "position"] = -1

    # 2. Calculate Strategy Returns
    test_df["strategy_return"] = test_df["position"].shift(1) * test_df["target_return"]

    # 3. Equity Curve
    test_df["strategy_equity"] = (
        initial_capital * (1 + test_df["strategy_return"].fillna(0)).cumprod()
    )
    test_df["buy_hold_equity"] = (
        initial_capital * (1 + test_df["target_return"].fillna(0)).cumprod()
    )

    # =========================================================
    # 📊 RESULTS & PLOTTING
    # =========================================================
    final_strat = test_df["strategy_equity"].iloc[-1]
    final_bh = test_df["buy_hold_equity"].iloc[-1]
    return_strat = (final_strat - initial_capital) / initial_capital * 100
    return_bh = (final_bh - initial_capital) / initial_capital * 100

    print("\n" + "=" * 40)
    print(f"💰 FINAL RESULTS ({len(test_df)} Trading Days)")
    print("=" * 40)
    print(f"Initial Capital : ${initial_capital:,.2f}")
    print(f"Strategy Final  : ${final_strat:,.2f} ({return_strat:+.2f}%)")
    print(f"Buy & Hold Final: ${final_bh:,.2f} ({return_bh:+.2f}%)")

    if final_strat > final_bh:
        print("🚀 SUCCESS: Model outperformed Buy & Hold!")
    else:
        print("📉 REALITY CHECK: Buy & Hold won this time.")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        test_df.index, test_df["buy_hold_equity"], label="Buy & Hold (AAPL)", alpha=0.6
    )
    plt.plot(
        test_df.index,
        test_df["strategy_equity"],
        label="Random Forest Strategy",
        linewidth=2,
        color="green",
    )
    plt.title(f"Backtest: AI Model vs. Buy & Hold (Last {len(test_df)} Days)")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    run_backtest()
