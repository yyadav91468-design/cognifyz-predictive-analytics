
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best_sales_model.joblib"
DATA_PATH = ROOT / "data" / "processed" / "daily_sales_features.csv"

client = TestClient(app)


def make_valid_features():
    """Build a valid request from a real processed dataset row."""
    artifact = joblib.load(MODEL_PATH)
    data = pd.read_csv(DATA_PATH)

    feature_columns = artifact["feature_columns"]
    row = data.iloc[-1]

    return {
        feature: float(row[feature])
        for feature in feature_columns
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["status"] == "healthy"


def test_model_info_endpoint():
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()
    assert body["model_name"] == "Random Forest"
    assert "sales_lag_0" in body["feature_columns"]
    assert "target_next_day_sales" not in body["feature_columns"]


def test_prediction_returns_numeric_value():
    response = client.post(
        "/predict",
        json={"features": make_valid_features()},
    )

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["predicted_next_day_sales"], (int, float))
    assert body["currency"] == "GBP"
    assert body["model_name"] == "Random Forest"


def test_prediction_rejects_missing_features():
    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_prediction_rejects_unexpected_features():
    features = make_valid_features()
    features["unexpected_feature"] = 123.0

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422
    assert "unexpected_features" in response.json()["detail"]
