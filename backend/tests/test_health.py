"""Tests for health, ready, and metadata endpoints."""

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.config import SUPPORTED_LETTERS, EXCLUDED_LETTERS

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["service"] == "SIGNAL ASL Gesture Recognition Service"


def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert data["classes_count"] == 24
    assert len(data["supported_letters"]) == 24
    assert "A" in data["supported_letters"]
    assert "Y" in data["supported_letters"]
    # Check that J and Z are excluded
    assert "J" not in data["supported_letters"]
    assert "Z" not in data["supported_letters"]
    assert set(data["excluded_letters"]) == {"J", "Z"}
    assert data["input_dimension"] == 63
    assert data["preprocessing"]["version"] == "v2.0.0"


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    assert "model_loaded" in data
    assert data["classes_count"] == 24
