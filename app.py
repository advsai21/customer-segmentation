"""Streamlit dashboard for customer segmentation."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_validation import validate_customer_data
from src.clustering import fit_clustering
from src.evaluation import evaluate_clustering, compare_k_values

st.set_page_config(page_title="Customer Segmentation", page_icon="🛍️", layout="wide")
st.title("Customer Segmentation — Live")
st.caption("Explore customer groups using K-Means clustering.")

@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(file_bytes))

default_path = Path(os.getenv("CUSTOMER_DATA_PATH", "data/sample_customers.csv"))
uploaded = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
if uploaded is not None:
    raw = load_csv(uploaded.getvalue())
elif default_path.exists():
    raw = pd.read_csv(default_path)
    st.sidebar.caption(f"Using `{default_path}`")
else:
    st.error("No dataset found. Upload a CSV or add data/sample_customers.csv.")
    st.stop()

numeric_cols = raw.select_dtypes(include="number").columns.tolist()
if len(numeric_cols) < 1:
    st.error("Dataset must contain at least one numeric feature.")
    st.stop()

st.sidebar.header("Model configuration")
features = st.sidebar.multiselect("Features", numeric_cols, default=numeric_cols[:2] or numeric_cols[:1])
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=min(10, max(2, len(raw)-1)), value=min(5, max(2, len(raw)-1)))
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)

if len(raw) <= k:
    st.warning("Please upload more rows than the selected number of clusters.")
    st.stop()
if not features:
    st.info("Choose at least one numeric feature in the sidebar.")
    st.stop()

try:
    X = validate_customer_data(raw, features, min_rows=k + 1)
    model = fit_clustering(X, k=int(k), random_state=int(seed))
    labels = model.predict(X)
    metrics = evaluate_clustering(model, X)
except (ValueError, TypeError) as exc:
    st.error(str(exc))
    st.stop()

result = raw.copy()
result["Cluster"] = labels
a, b, c, d = st.columns(4)
a.metric("Customers", f"{len(result):,}")
b.metric("Clusters", metrics["n_clusters"])
c.metric("Silhouette score", "N/A" if metrics["silhouette_score"] is None else f'{metrics["silhouette_score"]:.3f}')
d.metric("Davies–Bouldin", "N/A" if metrics["davies_bouldin_score"] is None else f'{metrics["davies_bouldin_score"]:.3f}')

st.subheader("Customer segments")
if len(features) >= 2:
    fig = px.scatter(result, x=features[0], y=features[1], color=result["Cluster"].astype(str),
                     hover_data=[c for c in raw.columns if c not in features],
                     title=f"Clusters by {features[0]} and {features[1]}")
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = px.histogram(result, x=features[0], color=result["Cluster"].astype(str),
                       barmode="overlay", title=f"Distribution by {features[0]}")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Cluster profiles")
profile = result.groupby("Cluster")[features].mean().round(2)
profile.insert(0, "Customer count", result.groupby("Cluster").size())
st.dataframe(profile, use_container_width=True)

with st.expander("Compare K values"):
    max_k = min(9, len(X)-1)
    if max_k >= 2:
        comparison = compare_k_values(X, range(2, max_k+1), random_state=int(seed))
        st.dataframe(comparison.round(4), use_container_width=True)
        if not comparison.empty:
            st.line_chart(comparison.set_index("k")[["inertia", "silhouette_score"]])
    else:
        st.info("More rows are needed to compare cluster counts.")

st.subheader("Assign a new customer")
with st.form("new_customer"):
    values = {}
    cols = st.columns(min(3, len(features)))
    for i, feature in enumerate(features):
        with cols[i % len(cols)]:
            values[feature] = st.number_input(feature, value=float(X[feature].median()))
    submitted = st.form_submit_button("Predict segment")
if submitted:
    new_row = pd.DataFrame([values], columns=features)
    segment = int(model.predict(new_row)[0])
    st.success(f"This customer is assigned to Cluster {segment}.")

st.subheader("Segmented data")
st.dataframe(result, use_container_width=True)
csv = result.to_csv(index=False).encode("utf-8")
st.download_button("Download segmented CSV", data=csv, file_name="segmented_customers.csv", mime="text/csv")
