"""Feature preprocessing pipeline."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_preprocessor() -> Pipeline:
    """Create a fitted-at-training-time standardization step."""
    return Pipeline([("scaler", StandardScaler())])
