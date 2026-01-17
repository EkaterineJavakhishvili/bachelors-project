import pandas as pd


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames columns to enforce Naming Conventions (Title Case).
    Example: 'close' -> 'Close', 'date' -> 'Date'
    """
    df = df.rename(
        columns={
            "date": "Date",
            "close": "Close",
            "high": "High",
            "low": "Low",
            "open": "Open",
            "volume": "Volume",
        }
    )

    # Ensure Date is available as a column if it's currently an index
    if "Date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        df = df.rename(columns={"index": "Date"})

    return df
