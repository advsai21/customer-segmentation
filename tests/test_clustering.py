import numpy as np
import pytest
from src.clustering import fit_clustering, make_clustering_pipeline
from src.data_validation import validate_customer_data
from src.evaluation import evaluate_clustering

def test_pipeline_fits_and_predicts(customer_df):
    X = validate_customer_data(customer_df, ["income", "spending"])
    model = fit_clustering(X, k=3, random_state=7)
    predictions = model.predict(X)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({0, 1, 2})

def test_k_greater_than_rows_fails(customer_df):
    with pytest.raises(ValueError, match="must be >= k"):
        fit_clustering(customer_df, k=len(customer_df)+1)

def test_invalid_k_fails():
    with pytest.raises(ValueError, match="k must"):
        make_clustering_pipeline(k=0)

def test_evaluation_metrics_are_valid(customer_df):
    X = validate_customer_data(customer_df, ["income", "spending"])
    model = fit_clustering(X, k=3)
    metrics = evaluate_clustering(model, X)
    assert metrics["n_samples"] == len(X)
    assert metrics["n_clusters"] >= 2
    assert metrics["inertia"] >= 0
    assert -1 <= metrics["silhouette_score"] <= 1
    assert metrics["davies_bouldin_score"] >= 0
