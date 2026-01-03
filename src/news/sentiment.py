import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from src.config import DATA_DIR

# Industry-standard Financial BERT model
MODEL_NAME = "ProsusAI/finbert"


def load_finbert():
    print(f"⏳ Loading FinBERT model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


def get_sentiment_score(headlines, tokenizer, model):
    """
    Scoring a batch of headlines.
    Returns a scalar: +1 (Positive) to -1 (Negative).
    """
    if not headlines:
        return 0.0

    # Tokenize
    inputs = tokenizer(
        headlines, return_tensors="pt", padding=True, truncation=True, max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)

    # FinBERT Output Mapping: 0=Positive, 1=Negative, 2=Neutral
    # We convert this to a single signal: (Positive * 1) + (Negative * -1)
    sentiment_vals = (scores[:, 0] * 1) + (scores[:, 1] * -1)

    return sentiment_vals.mean().item()


def process_news():
    # 1. Load the Kaggle CSV
    news_path = DATA_DIR / "raw" / "news" / "apple_news_data.csv"

    if not news_path.exists():
        print(f"❌ Error: File not found at {news_path}")
        return

    print(f"Loading news from {news_path}...")
    # on_bad_lines='skip' handles formatting errors in the CSV
    df = pd.read_csv(news_path, on_bad_lines="skip")

    # 2. Clean Dates (Handling ISO format with Timezone T...Z)
    # utc=True is required because your data has "+00:00"
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date

    # Clean rows
    df = df.dropna(subset=["date", "title"])
    df = df.sort_values("date")

    print(f"✅ Data Loaded. Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total Headlines: {len(df)}")

    # 3. Setup FinBERT
    tokenizer, model = load_finbert()

    print("🧠 Scoring headlines with FinBERT (This is the slow part)...")

    # Group by Date
    daily_groups = df.groupby("date")["title"].apply(list)
    results = []

    # 4. Inference Loop
    for date, headlines in tqdm(daily_groups.items(), total=len(daily_groups)):
        # Optimization: Take top 20 headlines/day to speed up processing
        daily_score = get_sentiment_score(headlines[:20], tokenizer, model)
        results.append({"date": date, "sentiment_finbert": daily_score})

    # 5. Save
    sentiment_df = pd.DataFrame(results)
    sentiment_df["date"] = pd.to_datetime(
        sentiment_df["date"]
    )  # Ensure datetime format for merging
    sentiment_df = sentiment_df.set_index("date")

    out_path = DATA_DIR / "processed" / "news" / "AAPL_sentiment.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sentiment_df.to_parquet(out_path)

    print(f"✅ Sentiment Signal Saved -> {out_path}")
    print(sentiment_df.tail())


if __name__ == "__main__":
    process_news()
