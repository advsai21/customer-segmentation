"""K-Means model construction and fitting."""
from __future__ import annotations
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from .preprocessing import make_preprocessor

def make_clustering_pipeline(k: int = 5, random_state: int = 42, n_init: int = 10) -> Pipeline:
    if k < 1:
        raise ValueError("k must be at least 1.")
    if n_init < 1:
        raise ValueError("n_init must be at least 1.")
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("kmeans", KMeans(n_clusters=k, random_state=random_state, n_init=n_init)),
    ])

def fit_clustering(df: pd.DataFrame, k: int = 5, random_state: int = 42) -> Pipeline:
    if len(df) < k:
        raise ValueError(f"Number of rows ({len(df)}) must be >= k ({k}).")
    model = make_clustering_pipeline(k=k, random_state=random_state)
    model.fit(df)
    return model
