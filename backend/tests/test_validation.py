"""Tests for input validation and error handling."""

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_invalid_length_too_few():
    # 60 coordinates instead of 63
    payload = {"landmarks": [0.1] * 60}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "Expected exactly 63" in str(data)


def test_invalid_length_too_many():
    # 64 coordinates instead of 63
    payload = {"landmarks": [0.1] * 64}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_nan_values_rejected():
    landmarks = [0.1] * 63
    landmarks[10] = None  # None turns into null in JSON
    payload = {"landmarks": landmarks}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_extreme_outlier_values_rejected():
    landmarks = [0.1] * 63
    landmarks[5] = 999.0  # Normalized coordinates should never be 999
    payload = {"landmarks": landmarks}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
