import pandas as pd
import pytest
from src.clustering import fit_clustering
from src.predict import save_model, load_model, assign_segments

def test_model_save_load_and_predict(customer_df, tmp_path):
    model = fit_clustering(customer_df, k=3)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    loaded = load_model(path)
    predictions = assign_segments(loaded, customer_df)
    assert len(predictions) == len(customer_df)
    assert all(isinstance(x, int) for x in predictions)

def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")
