from pathlib import Path
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()  # Values in .env file will be environmental variables (Anything becomes available via os.getenv)


# Directories for the pipeline
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PRICES = DATA_DIR / "raw" / "prices"
INT_PRICES = DATA_DIR / "interim" / "prices"
PROC_PRICES = DATA_DIR / "processed" / "prices"
MODELS_DIR = PROJECT_ROOT / "models" / "price_agent"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"


# Double checking if the folders exist
for d in [RAW_PRICES, INT_PRICES, PROC_PRICES, MODELS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


#  If .env doesn't have values, use the defaults below
TICKER = os.getenv("TICKER", "NFLX")  # Stock symbol we use
START = os.getenv("PRICES_START", "2018-01-01")  # Date range for hist price data
END = os.getenv("PRICES_END", str(date.today()))  # Date range for hist price data
LOOKBACK = int(os.getenv("LOOCKBACK", "60"))  # How many past days LSTM looks to predict
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.2"))  # 20% test set
VAL_RATIO = float(
    os.getenv("VAL_RATIO", "0.125")
)  # Validation (10-15% from training set)
