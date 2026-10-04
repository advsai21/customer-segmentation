"""Inference helpers for saved customer segmentation pipelines."""
from __future__ import annotations
import joblib
import pandas as pd
from pathlib import Path

def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}. Train the model first.")
    return joblib.load(path)

def assign_segments(model, customers: pd.DataFrame) -> list[int]:
    return model.predict(customers).astype(int).tolist()
