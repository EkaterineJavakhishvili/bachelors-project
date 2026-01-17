import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from src.utils import load_data
from src.config import FIG_DIR


def run_backtest(
    ticker: str = "AAPL", initial_capital: float = 10000.0, end_date: str = "2024-11-27"
) -> None:
    """
    Executes backtest simulation using Random Forest model.

    Strategy:
    - Trains on past data
    - Predicts next day's return
    - Long (Buy) if Prediction > 0.1%
    - Cash (Neutral) otherwise

    Args:
        ticket (str): Stock symbol to backtest (Defaults to "AAPL").
        initial_capital (float): Starting portfolio value (Defaults to $10,000).
    """
    print(f"\n🚀 STARTING BACKTEST SIMULATION FOR {ticker}...")

    try:
        # 1. load data using shared utility
        # this automatically handles finding the file in the correct folder
        df = load_data("features", f"{ticker}_features.parquet")

        # Cut off data where News ends
        if end_date:
            print(f"✂️  Trimming simulation at {end_date} (Matching News Data)...")
            cutoff_dt = pd.to_datetime(end_date).date()
            df = df.loc[:cutoff_dt]

        # critical check: ensure having target variable
        if "target_return" not in df.columns:
            raise ValueError(
                f"❌ Critical: '{ticker}' data is missing 'target_return' column. Run features pipeline first."
            )

        # 2. split data (80% train, 20% test)
        # using time-series split to avoid looking into the future
        test_size = int(len(df) * 0.2)
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:].copy()

        print(f"   Test Date Range : {test_df.index[0]} to {test_df.index[-1]}")

        # filter feature columns (exclude dates and non-numeric targets)
        feature_cols = [
            c for c in df.columns if c not in ["target_return", "close", "date", "Date"]
        ]
        print(f"   Using Features: {feature_cols}")

        # 3. train model
        print(f"🤖 Training Random Forest on {len(train_df)} historical days...")
        # balanced: depth 10 + leaf 2 (not overfitting single days)
        model = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42
        )
        model.fit(train_df[feature_cols], train_df["target_return"])

        # 4. generate signals
        print("⚡ Generating trading signals...")
        test_df["pred_return"] = model.predict(test_df[feature_cols])

        # strategy logic: buy if model predicts > 0.1% return
        test_df["position"] = 0
        test_df.loc[test_df["pred_return"] > 0.001, "position"] = 1

        # 5. calculate performance
        # shifting position by 1 because trading based on today's signal for tomorrow's return
        test_df["strategy_return"] = (
            test_df["position"].shift(1) * test_df["target_return"]
        )

        # calculate equity curves
        test_df["strategy_equity"] = (
            initial_capital * (1 + test_df["strategy_return"].fillna(0)).cumprod()
        )
        test_df["buy_hold_equity"] = (
            initial_capital * (1 + test_df["target_return"].fillna(0)).cumprod()
        )

        # 6. report results
        final_strat = test_df["strategy_equity"].iloc[-1]
        final_bh = test_df["buy_hold_equity"].iloc[-1]
        return_pct = ((final_strat - initial_capital) / initial_capital) * 100

        print("\n" + "=" * 40)
        print(f"💰 BACKTEST RESULTS ({ticker})")
        print("=" * 40)
        print(f"Initial Capital : ${initial_capital:,.2f}")
        print(f"Strategy Final  : ${final_strat:,.2f} ({return_pct:.2f}%)")
        print(f"Buy & Hold      : ${final_bh:,.2f}")
        print("=" * 40)

        # 7. visualization
        plt.figure(figsize=(12, 6))
        plt.plot(
            test_df.index,
            test_df["buy_hold_equity"],
            label="Buy & Hold (Benchmark)",
            alpha=0.5,
            linestyle="--",
        )
        plt.plot(
            test_df.index,
            test_df["strategy_equity"],
            label="AI Strategy",
            linewidth=2,
            color="green",
        )
        plt.title(f"AI Strategy vs Market: {ticker}")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_img = FIG_DIR / "backtest_result.png"
        out_img.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_img)
        print(f"🖼️  Plot saved to: {out_img}")

    except Exception as e:
        print(f"❌ ERROR in Backtest: {e}")


if __name__ == "__main__":
    # parsing command line arguments for flexibility
    parser = argparse.ArgumentParser(description="Run backtest simulation.")

    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock symbol to backtest (default: AAPL)",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital in USD (default: 10000.0)",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Optional cutoff date (YYYY-MM-DD) to stop the simulation",
    )

    args = parser.parse_args()

    run_backtest(
        ticker=args.ticker, initial_capital=args.capital, end_date=args.end_date
    )
