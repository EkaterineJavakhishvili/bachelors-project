import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from src.config import DATA_DIR

# industry standard Financial BERT model
MODEL_NAME = "ProsusAI/finbert"


def get_device():
    """
    Auto detects the best available hardware accelerator.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Mac M1/M2/M3
    elif torch.cuda.is_available():
        return torch.device("cuda")  # NVIDIA GPU
    else:
        return torch.device("cpu")  # Standard CPU


def load_finbert(device):
    """
    Loads the FinBERT model and moves it to the accelerator
    """
    print(f"⏳ Loading FinBERT model ({device})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


def process_news(ticker: str = "AAPL", input_file: str = "apple_news_data.csv"):
    """
    Main pipeline: Loads CSV -> Runs FinBERT -> Saves Daily Sentiment.
    """

    # 1. setup device
    device = get_device()
    print(f"🚀 Acceleration Status: Running on {device.type.upper()}")

    # 2. load data
    raw_news_path = DATA_DIR / "raw" / "news" / input_file

    if not raw_news_path.exists():
        print(f"❌ Error: File not found at {raw_news_path}")
        print("   Please ensure your dataset is in 'data/raw/news/'")
        return

    print(f"📂 Loading data from {raw_news_path.name}...")
    df = pd.read_csv(raw_news_path, on_bad_lines="skip")

    # 3. cleaning dates (Handling ISO formats and timezones standardizes the join key)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date", "title"])
    df = df.sort_values("date")

    print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Total Headlines: {len(df):,}")

    # 4. initialize model
    tokenizer, model = load_finbert(device)

    # 5. group by date
    print("🧠 Analyzing sentiment (this may take time)...")

    daily_groups = df.groupby("date")["title"].apply(list)
    results = []

    for date, headlines in tqdm(
        daily_groups.items(), total=len(daily_groups), unit="day"
    ):
        # optimization: limit to top 20 headlines per day to prevent memory crashes
        batch = headlines[:20]

        if not batch:
            results.append({"date": date, "sentiment_finbert": 0.0})
            continue

        # tokenize
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=64
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # softmax to get probabilities (0-1)
            scores = F.softmax(outputs.logits, dim=1)

        # FinBERT Labels: [Positive, Negative, Neutral] (Standard for ProsusAI)
        # Score = (Prob_Pos * 1) + (Prob_Neg * -1)
        # Neutral contributes 0.
        sentiment_vals = (scores[:, 0] * 1) + (scores[:, 1] * -1)

        # average sentiment for the day
        daily_score = sentiment_vals.mean().item()
        results.append({"date": date, "sentiment_finbert": daily_score})

        # 6. save results
        sent_df = pd.DataFrame(results)

        sent_df["date"] = pd.to_datetime(sent_df["date"])
        sent_df = sent_df.set_index("date")

        out_path = DATA_DIR / "processed" / "news" / f"{ticker}_sentiment.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sent_df.to_parquet(out_path)

        print(f"✅ SUCCESS: Sentiment data saved to {out_path}")
        print(sent_df.tail())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Sentiment Features using FinBERT."
    )

    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Ticker symbol for the output filename (e.g., AAPL)",
    )

    parser.add_argument(
        "--file",
        type=str,
        default="apple_news_data.csv",
        help="Filename of the raw CSV in data/raw/news/",
    )

    args = parser.parse_args()

    process_news(ticker=args.ticker, input_file=args.file)
