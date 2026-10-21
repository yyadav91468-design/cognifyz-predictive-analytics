
# Sales Prediction & Business Intelligence Dashboard

**Cognifyz Technologies Internship — Level 3: Advanced**

A predictive analytics project that processes retail transaction data, compares machine-learning regression models, exposes predictions through FastAPI, and presents sales analytics in an interactive Streamlit dashboard.

## Features

- Retail transaction data cleaning and validation
- Daily sales aggregation and feature engineering
- Calendar, lag, and rolling-window features
- Chronological training and test split
- Regression model comparison using MAE, RMSE, and R²
- Saved model artifact with feature schema and evaluation metrics
- FastAPI prediction endpoint and interactive Swagger documentation
- Streamlit dashboard with historical trends, evaluation charts, and prediction form
- Automated API tests

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- FastAPI and Uvicorn
- Streamlit and Plotly
- Joblib
- pytest and HTTPX

## Dataset

This project uses the UCI Online Retail dataset:

https://archive.ics.uci.edu/dataset/352/online-retail

Place the downloaded Excel file at:

`data/raw/Online Retail.xlsx`

The raw data contains transaction-level records. The modelling pipeline aggregates valid transactions into daily sales and creates features to estimate next-day sales.

## Project Structure

```text
cognevance_predictive_analytics/
├── api/
│   └── main.py
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── MODEL_REPORT.md
├── models/
│   └── best_sales_model.joblib
├── reports/
│   ├── data_quality_report.txt
│   ├── model_comparison.csv
│   ├── model_evaluation.txt
│   └── test_predictions.csv
├── src/
│   ├── preprocess_data.py
│   └── train_model.py
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Windows Setup

Open PowerShell in the project root.

Create and activate a virtual environment if one does not already exist:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the dataset and place it at `data/raw/Online Retail.xlsx`.

## Run the Pipeline

Run preprocessing first:

```powershell
python -m src.preprocess_data
```

Train and evaluate the models:

```powershell
python -m src.train_model
```

The training script saves the selected model artifact and evaluation outputs. Retrain after changing preprocessing or feature definitions.

## Start the API

From the project root:

```powershell
python -m uvicorn api.main:app --reload
```

Open these local URLs:

- API health: http://127.0.0.1:8000/health
- Model schema and metrics: http://127.0.0.1:8000/model-info
- Swagger API documentation: http://127.0.0.1:8000/docs

The prediction endpoint is `POST /predict`. It expects every numeric feature listed by `/model-info`.

## Start the Dashboard

Open a second PowerShell terminal, activate the virtual environment, and run:

```powershell
python -m streamlit run dashboard/app.py
```

Open http://localhost:8501.

## Run Automated Tests

```powershell
python -m pytest tests/ -v
```

Tests require the processed dataset and trained model artifact to exist.

## Evaluation

The initial chronological holdout evaluation selected Random Forest:

| Metric | Result |
|---|---:|
| MAE | 16,102.55 |
| RMSE | 27,008.60 |
| R² | 0.2543 |

These are the current recorded results. Rerun training to regenerate the reports if the code or dataset changes.

## Limitations

- The prepared modelling dataset contains only a few hundred daily observations, even though the raw dataset has hundreds of thousands of transactions.
- The model's predictive performance is modest and may not generalize to other retailers or future periods.
- Predictions are estimates, not guarantees of future revenue.
- Currency and historical values follow the dataset and modelling configuration; they should not be interpreted as current commercial performance.

## Further Improvements

- Compare against seasonal-naive and time-series forecasting baselines.
- Use rolling-origin time-series validation.
- Tune model parameters using time-aware validation.
- Add prediction intervals and drift monitoring.
- Deploy the API and dashboard with secure configuration and monitoring.

## Project Status

Local preprocessing, model training, FastAPI prediction, and Streamlit dashboard have been run successfully during development. Confirm automated tests and any cloud deployment independently before claiming those stages as complete.
