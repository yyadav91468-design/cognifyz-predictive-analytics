
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "daily_sales_features.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

TARGET = "target_next_day_sales"

FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "day_of_year",
    "is_weekend",
    "sales_lag_0",
    "sales_lag_1",
    "sales_lag_2",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
]


def load_dataset():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_FILE}\n"
            "Run: python -m src.preprocess_data"
        )

    df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    missing = set(FEATURES + [TARGET, "Date"]) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}. "
            "Rerun preprocessing using the updated script."
        )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + [TARGET])

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} usable rows found. "
            "At least 50 are required for this baseline."
        )

    return df


def evaluate(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    # Chronological split: earlier observations train the model,
    # while the latest 20% of observations are held out for testing.
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    models = {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    results = []
    trained_models = {}

    print(f"Total usable rows: {len(df):,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print(
        f"Training period: {train_df['Date'].min().date()} "
        f"to {train_df['Date'].max().date()}"
    )
    print(
        f"Testing period: {test_df['Date'].min().date()} "
        f"to {test_df['Date'].max().date()}"
    )

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        scores = evaluate(y_test, predictions)

        results.append({
            "Model": name,
            "MAE": scores["MAE"],
            "RMSE": scores["RMSE"],
            "R2": scores["R2"],
        })

        trained_models[name] = model

        print(
            f"MAE: {scores['MAE']:.2f} | "
            f"RMSE: {scores['RMSE']:.2f} | "
            f"R2: {scores['R2']:.4f}"
        )

    comparison = pd.DataFrame(results).sort_values("RMSE")
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    # Select the best model using the predeclared RMSE metric.
    best_name = comparison.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_predictions = best_model.predict(X_test)

    best_rmse = float(
        np.sqrt(mean_squared_error(y_test, best_predictions))
    )
    best_mae = float(mean_absolute_error(y_test, best_predictions))
    best_r2 = float(r2_score(y_test, best_predictions))

    # Persist model and schema together so the API uses the same
    # feature order that was used during training.
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "test_metrics": {
            "MAE": best_mae,
            "RMSE": best_rmse,
            "R2": best_r2,
        },
        "training_rows": len(train_df),
        "test_rows": len(test_df),
        "test_start_date": str(test_df["Date"].min().date()),
        "test_end_date": str(test_df["Date"].max().date()),
        "currency": "GBP",
    }

    model_path = MODEL_DIR / "best_sales_model.joblib"
    joblib.dump(artifact, model_path)

    # Save predictions from the holdout period for the dashboard.
    prediction_report = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual_Sales": y_test.values,
        "Predicted_Sales": best_predictions,
        "Absolute_Error": np.abs(y_test.values - best_predictions),
    })
    prediction_report.to_csv(
        REPORT_DIR / "test_predictions.csv", index=False
    )

    # Save feature importance for tree-based models.
    if hasattr(best_model, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": best_model.feature_importances_,
        }).sort_values("Importance", ascending=False)

        importance.to_csv(
            REPORT_DIR / "feature_importance.csv", index=False
        )

    report_lines = [
        "COGNEVANCE - SALES PREDICTION MODEL REPORT",
        "=" * 52,
        f"Selected model: {best_name}",
        "Selection metric: lowest RMSE on chronological holdout",
        f"Training rows: {len(train_df)}",
        f"Testing rows: {len(test_df)}",
        f"Test period: {test_df['Date'].min().date()} to "
        f"{test_df['Date'].max().date()}",
        f"MAE: {best_mae:.4f}",
        f"RMSE: {best_rmse:.4f}",
        f"R2: {best_r2:.4f}",
        "",
        "MODEL COMPARISON",
        comparison.to_string(index=False),
        "",
        "Currency: GBP",
        "Metrics describe the holdout period only.",
        "This baseline does not guarantee future forecasting accuracy.",
    ]

    (REPORT_DIR / "model_evaluation.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print("\nMODEL COMPARISON")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {REPORT_DIR / 'model_evaluation.txt'}")
    print(f"Saved predictions: {REPORT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
