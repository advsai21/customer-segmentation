# Customer Segmentation with K-Means — CI/CD

An end-to-end, reproducible customer-segmentation project using Python, scikit-learn, pytest, GitHub Actions, Streamlit and optional Docker/Render deployment.

## Features
- Validates input data and required numerical features
- Fits preprocessing and K-Means together in a scikit-learn pipeline
- Evaluates clustering with silhouette score, Davies–Bouldin index and inertia
- Compares a configurable range of K values
- Saves a complete pipeline artifact with joblib
- Interactive Streamlit dashboard for exploration and assigning a new customer
- Unit tests and GitHub Actions CI on pushes and pull requests
- Optional CD workflow using a Render deploy hook

## Quick start (Python 3.10+)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run with the included sample data

```bash
python -m src.train --data data/sample_customers.csv --features Annual Income (k$) Spending Score (1-100) --k 5
streamlit run app.py
```

The training command writes `models/customer_kmeans.joblib` and `reports/evaluation.json`. The included CSV is synthetic demo data and is not real customer information.

### Use your own data

Place a CSV at `data/customers.csv`, then specify its numeric feature columns:

```bash
python -m src.train --data data/customers.csv --features AnnualIncome SpendingScore --k 4
```

Feature names containing spaces are supported by passing each feature as a separate quoted argument, e.g. `--features "Annual Income" "Spending Score"`.

You can also use environment variables:
- `CUSTOMER_DATA_PATH`: default dataset path used by the dashboard
- `MODEL_PATH`: model artifact path (default `models/customer_kmeans.joblib`)

The app uses the included sample dataset if the configured dataset is absent. Upload a CSV in the sidebar to explore another dataset. Select the same features used to train the model for valid predictions.

## Project layout

```text
customer-segmentation/
├── app.py
├── src/
│   ├── data_validation.py
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── train.py
│   └── predict.py
├── tests/
├── data/sample_customers.csv
├── models/.gitkeep
├── reports/.gitkeep
├── .github/workflows/ci.yml
├── .github/workflows/cd.yml
├── Dockerfile
├── render.yaml
├── requirements.txt
└── .gitignore
```

## CI/CD

**CI** (`.github/workflows/ci.yml`) runs on pushes and pull requests to `main`/`master`: installs dependencies, compiles source, runs pytest and performs a small training smoke test.

**CD** (`.github/workflows/cd.yml`) runs after CI succeeds on a push to the default branch, or manually. To enable Render deployment:
1. Create a Render Web Service from this repository using the included `render.yaml` (or create a Python service manually).
2. Add a GitHub Actions repository secret named `RENDER_DEPLOY_HOOK` containing the service's deploy-hook URL.
3. Enable the `deploy` job in `cd.yml` by removing the `if: ${{ false }}` guard.
4. Push to the default branch. The workflow calls the deploy hook; Render builds and deploys the app.

The CD workflow is deliberately disabled by default so a fresh clone cannot accidentally trigger deployment. Render may charge for some service configurations; check its current plan and limits.

## Data and modeling notes
- K-Means is unsupervised: classification accuracy is not an appropriate default metric.
- Silhouette score is meaningful only when there are at least 2 clusters and fewer clusters than samples. The evaluation code handles invalid/small cases.
- Scaling matters because K-Means uses distance. `StandardScaler` is fitted inside the pipeline to avoid data leakage during future inference.
- Cluster IDs are arbitrary labels, not rankings or customer value judgments.
- Choose K using metrics, visualization and domain context rather than a single score alone.
- Do not commit confidential or personally identifiable customer data. The sample data is synthetic.

## Tests

```bash
pytest -v
```

## Docker

```bash
docker build -t customer-segmentation .
docker run --rm -p 8501:8501 customer-segmentation
```
Open `http://localhost:8501`.

## Troubleshooting
- **Missing columns:** confirm the feature names exactly match CSV headers.
- **Too few rows:** K-Means needs at least K samples; use more rows or reduce K.
- **Model not found:** train once with the command above; the dashboard can also train a model from its selected uploaded data.
- **Different clusters on each run:** the project sets a fixed random seed by default. Change `--random-state` intentionally to explore sensitivity.
