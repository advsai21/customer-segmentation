import pandas as pd
import pytest

@pytest.fixture
def customer_df():
    return pd.DataFrame({
        "income": [15, 18, 20, 75, 80, 85, 35, 38, 40, 90, 95, 100],
        "spending": [20, 25, 18, 80, 85, 78, 45, 50, 48, 15, 18, 12],
    })
