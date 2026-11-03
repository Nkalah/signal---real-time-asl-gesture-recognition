import json

import numpy as np
import torch

from backend.app.model import ASLLandmarkMLP
from backend.app.predictor import ASLPredictor
from training.calibration import calibrate_validation


def test_validation_calibration_returns_temperature_ece_and_rejection_thresholds():
    logits = np.asarray(
        [
            [5.0, 0.0, -1.0],
            [0.0, 5.0, -1.0],
            [0.0, 5.0, -1.0],
            [5.0, 0.0, -1.0],
            [-1.0, -1.0, 5.0],
            [5.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)

    calibration = calibrate_validation(logits, targets, ["A", "B", "NONE"])

    assert np.isfinite(calibration["temperature"])
    assert calibration["temperature"] > 0
    assert 0 <= calibration["ece_before"] <= 1
    assert 0 <= calibration["ece_after"] <= 1
    rejection = calibration["rejection"]
    assert 0 <= rejection["max_probability_below"] <= 1
    assert 0 <= rejection["top_two_margin_below"] <= 1
    assert rejection["none_label"] == "NONE"


def test_predictor_loads_calibrated_none_class_and_rejects_it(tmp_path):
    labels = ["A", "B", "NONE"]
    model = ASLLandmarkMLP(num_classes=len(labels))
    with torch.no_grad():
        model.fc3.weight.zero_()
        model.fc3.bias[:] = torch.tensor([0.0, 0.0, 10.0])
    weights_path = tmp_path / "weights.pth"
    config_path = tmp_path / "config.json"
    torch.save(model.state_dict(), weights_path)
    config_path.write_text(
        json.dumps(
            {
                "labels": labels,
                "version": "2.0.0",
                "training_metadata": {"dataset_source": "collected"},
                "calibration": {
                    "temperature": 1.5,
                    "rejection": {
                        "max_probability_below": 0.1,
                        "top_two_margin_below": 0.1,
                        "none_label": "NONE",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    predictor = ASLPredictor(weights_path=weights_path, config_path=config_path)
    prediction = predictor.predict([0.0] * 63)

    assert predictor.is_ready
    assert predictor.labels == labels
    assert predictor.temperature == 1.5
    assert prediction.letter is None
    assert prediction.probabilities["NONE"] > 0.99
