"""Command-line training entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .data_validation import validate_customer_data
from .clustering import fit_clustering
from .evaluation import evaluate_clustering, compare_k_values
from .predict import save_model

def train(data_path: str, features: list[str], k: int = 5, random_state: int = 42,
          model_path: str = "models/customer_kmeans.joblib",
          report_path: str = "reports/evaluation.json") -> dict:
    raw = pd.read_csv(data_path)
    clean = validate_customer_data(raw, features, min_rows=max(2, k))
    model = fit_clustering(clean, k=k, random_state=random_state)
    metrics = evaluate_clustering(model, clean)
    save_model(model, model_path)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"features": features, "random_state": random_state, **metrics}, f, indent=2)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train a K-Means customer segmentation model.")
    parser.add_argument("--data", default="data/sample_customers.csv")
    parser.add_argument("--features", nargs="+", default=["Annual Income (k$)", "Spending Score (1-100)"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-path", default="models/customer_kmeans.joblib")
    parser.add_argument("--report-path", default="reports/evaluation.json")
    args = parser.parse_args()
    metrics = train(args.data, args.features, args.k, args.random_state, args.model_path, args.report_path)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
