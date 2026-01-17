import pandas as pd
from pathlib import Path
from src.config import DATA_DIR


def load_data(subfolder: str, filename: str) -> pd.DataFrame:
    """
    Function to load data from the processed or raw directory.

    Args:
        subfolder (str): The directory inside data/ (e.g., 'features', 'prices').
        filename (str): The full filename (e.g., 'AAPL.csv').

    Returns:
        pd.DataFrame: Loaded data.

    Raises:
        FileNotFounfError: If the file is missing in both raw and processed.
    """

    # Try processed path first
    file_path = DATA_DIR / "processed" / subfolder / filename

    # If not found fallback to raw path
    if not file_path.exists():
        file_path = DATA_DIR / "raw" / subfolder / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"❌ Critical Error: Data file '{filename}' not found in {subfolder}."
        )

    print(f"🔍 Loading data from: {file_path}")

    if filename.endswith(".parquet"):
        return pd.read_parquet(file_path)
    elif filename.endswith(".csv"):
        return pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")
