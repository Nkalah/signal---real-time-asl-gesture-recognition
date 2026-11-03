"""Tests for model prediction endpoint and inference logic."""

import numpy as np
import torch
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.predictor import predictor
from backend.app.config import SUPPORTED_LETTERS
from training.prepare_data import generate_canonical_hand, normalize_landmarks

client = TestClient(app)


def test_prediction_canonical_hands():
    # Test letter 'A'
    hand_a = generate_canonical_hand("A")
    norm_a = normalize_landmarks(hand_a).tolist()
    
    response = client.post("/predict", json={"landmarks": norm_a})
    assert response.status_code == 200
    data = response.json()
    
    assert data["ready"] is True
    assert data["letter"] == "A"
    assert data["letter"] in SUPPORTED_LETTERS
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["inference_time_ms"] >= 0.0
    assert data["model_forward_time_ms"] >= 0.0
    assert data["top_predictions"] is not None
    assert len(data["top_predictions"]) >= 1
    
    # Check sum of probabilities equals ~1.0
    if data["probabilities"]:
        prob_sum = sum(data["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01


def test_model_predicts_all_canonical_training_templates():
    torch.set_num_threads(1)
    assert predictor.model.training is False
    X = torch.from_numpy(
        np.stack([
            normalize_landmarks(generate_canonical_hand(letter))
            for letter in SUPPORTED_LETTERS
        ])
    )
    expected = torch.arange(len(SUPPORTED_LETTERS))

    with torch.no_grad():
        predictions = predictor.model(X).argmax(dim=1)

    assert torch.equal(predictions, expected)


def test_prediction_structure_and_types():
    # Test letter 'L'
    hand_l = generate_canonical_hand("L")
    norm_l = normalize_landmarks(hand_l).tolist()
    
    response = client.post("/predict", json={"landmarks": norm_l})
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data["letter"], str)
    assert data["letter"] == "L"  # Canonical L should be recognized as L
    assert data["confidence"] > 0.80
    assert data["model_version"] == "1.0.0"
    assert data["error"] is None
