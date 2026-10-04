import pandas as pd
import pytest
from src.data_validation import validate_customer_data

def test_validation_returns_selected_numeric_columns(customer_df):
    result = validate_customer_data(customer_df, ["income", "spending"])
    assert list(result.columns) == ["income", "spending"]
    assert len(result) == len(customer_df)

def test_missing_feature_fails(customer_df):
    with pytest.raises(ValueError, match="Missing required"):
        validate_customer_data(customer_df, ["unknown"])

def test_missing_values_fail(customer_df):
    df = customer_df.copy()
    df.loc[0, "income"] = None
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_customer_data(df, ["income", "spending"])

def test_non_numeric_fails():
    df = pd.DataFrame({"income": ["low", "high"], "spending": [2, 3]})
    with pytest.raises(ValueError, match="missing or non-numeric"):
        validate_customer_data(df, ["income", "spending"])

def test_too_few_rows_fails():
    with pytest.raises(ValueError, match="At least"):
        validate_customer_data(pd.DataFrame({"x": [1]}), ["x"])
