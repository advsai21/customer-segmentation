"""Clustering metrics and K selection helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(model, df: pd.DataFrame) -> dict:
    transformed = model.named_steps["preprocessor"].transform(df)
    labels = model.named_steps["kmeans"].labels_
    n_clusters = len(np.unique(labels))
    result = {
        "n_samples": int(len(df)),
        "n_features": int(df.shape[1]),
        "n_clusters": int(n_clusters),
        "inertia": float(model.named_steps["kmeans"].inertia_),
        "silhouette_score": None,
        "davies_bouldin_score": None,
    }
    if 1 < n_clusters < len(df):
        result["silhouette_score"] = float(silhouette_score(transformed, labels))
        result["davies_bouldin_score"] = float(davies_bouldin_score(transformed, labels))
    return result

def compare_k_values(df: pd.DataFrame, k_values=range(2, 9), random_state: int = 42) -> pd.DataFrame:
    from .clustering import fit_clustering
    rows = []
    for k in k_values:
        if k < 2 or k >= len(df):
            continue
        model = fit_clustering(df, k=k, random_state=random_state)
        metrics = evaluate_clustering(model, df)
        rows.append({"k": k, **metrics})
    return pd.DataFrame(rows)
