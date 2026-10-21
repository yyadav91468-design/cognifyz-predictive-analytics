
# System Architecture

## Overview

The application separates data processing, model training, API inference, and dashboard presentation.

## Components

1. **Raw data** — UCI Online Retail Excel file stored under `data/raw/`.
2. **Preprocessing** — `src/preprocess_data.py` validates transactions, removes invalid sales, calculates revenue, aggregates daily sales, and generates model features.
3. **Processed dataset** — daily sales feature table stored under `data/processed/`.
4. **Model training** — `src/train_model.py` creates a chronological train/test split, compares regression algorithms, evaluates holdout predictions, and saves the selected model.
5. **Model artifact** — `models/best_sales_model.joblib` stores the trained estimator, feature column order, model name, target metadata, and test metrics.
6. **Prediction API** — `api/main.py` loads the artifact and exposes health, model-information, and prediction endpoints.
7. **Analytics dashboard** — `dashboard/app.py` displays historical trends, model evaluation, and interactive predictions.
8. **Tests and reports** — `tests/` contains automated API checks; `reports/` stores data-quality and model-evaluation outputs.

## Workflow

```text
UCI Excel Dataset
       |
       v
Data Cleaning and Validation
       |
       v
Daily Sales Aggregation
       |
       v
Feature Engineering
       |
       v
Chronological Train/Test Split
       |
       v
Model Training and Comparison
       |
       v
Saved Model Artifact
       |
       +----------------------+
       |                      |
       v                      v
FastAPI Prediction API   Streamlit Dashboard
       |                      |
       v                      v
JSON Prediction          Charts and User Inputs
```

## API Contract

- `GET /` — service entry point and documentation paths.
- `GET /health` — reports whether the model artifact loaded.
- `GET /model-info` — returns the model name, feature schema, currency metadata, and saved metrics.
- `POST /predict` — validates the feature names, orders values according to the saved schema, and returns a next-day sales estimate.
- `GET /docs` — interactive Swagger documentation supplied by FastAPI.

## Model Input Consistency

The model artifact contains the feature column order used during training. The API uses that schema to build the inference dataframe. It rejects missing or unexpected feature names instead of silently sending an incompatible input to the estimator.

## Validation Strategy

The modelling script uses a chronological split, training on earlier observations and evaluating on later observations. This respects temporal order better than a random split, although a single holdout period is not sufficient to establish performance across all seasons or business conditions.

## Known Limitations

- Daily aggregation reduces the raw transaction count to a much smaller number of modelling rows.
- The dashboard and API are local development services until separately deployed.
- The API accepts caller-provided features; it does not automatically fetch live transactions.
- The current model's test R² is modest, so predictions require caution.
