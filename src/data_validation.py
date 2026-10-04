"""Input-data validation helpers."""
from __future__ import annotations
import pandas as pd

def validate_customer_data(df: pd.DataFrame, features: list[str], min_rows: int = 2) -> pd.DataFrame:
    """Return a clean copy or raise ValueError with actionable feedback."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if df.empty or len(df) < min_rows:
        raise ValueError(f"At least {min_rows} rows are required.")
    if not features:
        raise ValueError("Select at least one feature.")
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if len(set(features)) != len(features):
        raise ValueError("Feature list contains duplicates.")

    clean = df.loc[:, features].copy()
    for col in features:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([float("inf"), float("-inf")], pd.NA)
    if clean.isna().any().any():
        counts = clean.isna().sum()
        details = {k: int(v) for k, v in counts[counts > 0].items()}
        raise ValueError(f"Features contain missing or non-numeric values: {details}")
    if not all(pd.api.types.is_numeric_dtype(clean[c]) for c in features):
        raise ValueError("All selected features must be numeric.")
    if clean.nunique(dropna=False).max() == 0:
        raise ValueError("Selected features contain no usable variation.")
    return clean.reset_index(drop=True)
