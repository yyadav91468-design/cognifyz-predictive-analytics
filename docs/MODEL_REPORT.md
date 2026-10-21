
# Predictive Model Report

## Objective

Estimate next-day retail sales from historical daily sales and calendar features.

## Dataset

The project uses the UCI Online Retail dataset. The source file contains 541,909 transaction rows in the version used during development.

The preprocessing run reported:

- Cancellation rows excluded: 9,288
- Other invalid sales rows excluded: 2,517
- Valid positive-sales rows: 530,104
- Prepared feature dataset: 345 daily rows

The transaction-level row count and the daily modelling row count are different because transactions are aggregated by date and initial rows without sufficient lag history are excluded.

## Feature Engineering

The generated features include:

- Calendar: day of week, day of month, month, day of year, weekend flag
- Sales lags: current-day sales and selected historical lags
- Rolling statistics: trailing seven-day and fourteen-day means, and trailing seven-day standard deviation
- Target: next-day daily sales

Rolling statistics use shifted historical values to avoid incorporating the target day's sales into those calculations. The target column is excluded from model input features.

## Evaluation Design

The prepared rows are sorted by date. The earliest 80% are used for training and the latest 20% are reserved for testing.

- Training rows: 276
- Testing rows: 69
- Training period: 2010-12-29 to 2011-09-30
- Testing period: 2011-10-01 to 2011-12-08

## Model Comparison

Recorded holdout metrics:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 31,500.80 | 39,533.01 | -0.5976 |
| Linear Regression | 19,527.56 | 28,455.46 | 0.1723 |
| Random Forest | 16,102.55 | 27,008.60 | 0.2543 |
| Extra Trees | 17,476.99 | 28,882.71 | 0.1472 |
| Hist Gradient Boosting | 17,301.65 | 27,405.40 | 0.2322 |

Random Forest had the lowest RMSE among the tested models and was selected as the saved model.

## Interpretation

- MAE is the mean absolute difference between actual and predicted sales.
- RMSE penalizes large prediction errors more strongly than MAE.
- R² measures performance relative to predicting the test-set mean. A value of 0.2543 indicates that the model explains only a limited portion of the test-set variation.
- The mean baseline's negative R² indicates that its predictions performed worse than the test-set mean benchmark for this holdout period.

The results support a working predictive analytics prototype, not a high-confidence commercial forecast.

## Reproducibility

From the project root:

```powershell
python -m src.preprocess_data
python -m src.train_model
```

The scripts regenerate the processed dataset, model artifact, comparison report, evaluation report, and holdout predictions.

## Limitations and Future Work

- Only 345 daily observations are available for modelling.
- The holdout covers one short historical period from a single retailer dataset.
- No claim is made that the reported score generalizes to new retailers or future market conditions.
- Future work should compare seasonal-naive baselines, use rolling-origin validation, tune models using time-aware methods, and quantify forecast uncertainty.
