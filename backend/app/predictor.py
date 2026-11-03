"""Predictor module for ASL landmark inference."""

import time
import json
import math
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

from .config import (
    MODEL_WEIGHTS_PATH,
    MODEL_CONFIG_PATH,
    SUPPORTED_LETTERS,
    INPUT_FEATURES,
    NUM_CLASSES,
    PREPROCESSING_VERSION
)
from .model import ASLLandmarkMLP
from .schemas import PredictionResponse, TopPrediction


class ASLPredictor:
    """Singleton predictor that manages model loading and inference."""
    
    def __init__(self, weights_path: Path = MODEL_WEIGHTS_PATH, config_path: Path = MODEL_CONFIG_PATH):
        self.weights_path = weights_path
        self.config_path = config_path
        self.model: Optional[ASLLandmarkMLP] = None
        self.labels: List[str] = list(SUPPORTED_LETTERS)
        self.is_ready: bool = False
        self.device = torch.device("cpu")
        self.model_version: str = "1.0.0"
        self.training_data_source: str = "unknown"
        self.load_error: Optional[str] = None
        self.temperature: float = 1.0
        self.max_probability_threshold: float = 0.0
        self.top_two_margin_threshold: float = 0.0
        self.none_label: str = "NONE"
        
        # Attempt to load model at initialization
        self.load_model()
        
    def load_model(self) -> bool:
        """Loads model weights and config from disk."""
        try:
            if not self.weights_path.exists():
                self.load_error = f"Model artifact not found at {self.weights_path}. Please run training pipeline first."
                self.is_ready = False
                self.model = None
                return False
                
            config: Dict[str, Any] = {}
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            labels = config.get("labels", config.get("supported_letters", SUPPORTED_LETTERS))
            if (
                not isinstance(labels, list)
                or not labels
                or any(not isinstance(label, str) or not label for label in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError("Model config must contain a non-empty, unique string label list.")

            model = ASLLandmarkMLP(
                input_dim=INPUT_FEATURES,
                hidden_dim_1=128,
                hidden_dim_2=64,
                num_classes=len(labels),
            )
            
            # Load state dict
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.labels = labels
            self.is_ready = True
            self.load_error = None

            calibration = config.get("calibration", {})
            rejection = calibration.get("rejection", {})
            temperature = float(calibration.get("temperature", 1.0))
            confidence_threshold = float(rejection.get("max_probability_below", 0.0))
            margin_threshold = float(rejection.get("top_two_margin_below", 0.0))
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("Calibration temperature must be a positive finite number.")
            if (
                not math.isfinite(confidence_threshold)
                or not 0 <= confidence_threshold <= 1
                or not math.isfinite(margin_threshold)
                or not 0 <= margin_threshold <= 1
            ):
                raise ValueError("Rejection thresholds must be finite probabilities between 0 and 1.")
            self.temperature = temperature
            self.max_probability_threshold = confidence_threshold
            self.top_two_margin_threshold = margin_threshold
            self.none_label = str(rejection.get("none_label", "NONE"))
            self.model_version = str(config.get("version", self.model_version))
            self.training_data_source = config.get(
                "training_metadata", {}
            ).get("dataset_source", "unknown")
                    
            return True
            
        except Exception as e:
            self.load_error = f"Failed to load model weights: {str(e)}"
            self.is_ready = False
            self.model = None
            return False

    def predict(self, landmarks: List[float]) -> PredictionResponse:
        """Performs real-time inference on 63 normalized coordinates."""
        if not self.is_ready or self.model is None:
            return PredictionResponse(
                ready=False,
                letter=None,
                confidence=0.0,
                model_version=self.model_version,
                inference_time_ms=0.0,
                error=self.load_error or "Model is not loaded or ready for inference."
            )
            
        start_time = time.perf_counter()
        
        try:
            # Prepare tensor
            tensor_input = torch.tensor(landmarks, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Model inference with no gradients
            model_start = time.perf_counter()
            with torch.no_grad():
                logits = self.model(tensor_input)
            model_forward_ms = (time.perf_counter() - model_start) * 1000.0
            with torch.no_grad():
                probabilities = F.softmax(logits / self.temperature, dim=1).squeeze(0)
                
            # Top prediction
            conf, pred_idx = torch.max(probabilities, dim=0)
            best_idx = int(pred_idx.item())
            predicted_label = self.labels[best_idx]
            best_conf = float(conf.item())
            top_two = torch.topk(probabilities, k=min(2, len(self.labels))).values
            margin = float((top_two[0] - top_two[1]).item()) if len(top_two) > 1 else best_conf
            rejected = (
                best_conf < self.max_probability_threshold
                or margin < self.top_two_margin_threshold
                or predicted_label == self.none_label
            )
            best_letter = None if rejected else predicted_label
            
            # Top-3 predictions
            topk_confs, topk_indices = torch.topk(probabilities, k=min(3, len(self.labels)))
            top_predictions = [
                TopPrediction(
                    letter=self.labels[int(idx.item())],
                    confidence=float(c.item())
                )
                for c, idx in zip(topk_confs, topk_indices)
            ]
            
            # Full probabilities mapping
            probs_dict = {
                self.labels[i]: float(probabilities[i].item())
                for i in range(len(self.labels))
            }
            
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            
            return PredictionResponse(
                ready=True,
                letter=best_letter,
                confidence=best_conf,
                probabilities=probs_dict,
                top_predictions=top_predictions,
                model_version=self.model_version,
                inference_time_ms=round(duration_ms, 2),
                model_forward_time_ms=round(model_forward_ms, 2),
                error=None
            )
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return PredictionResponse(
                ready=True,
                letter=None,
                confidence=0.0,
                model_version=self.model_version,
                inference_time_ms=round(duration_ms, 2),
                model_forward_time_ms=round(duration_ms, 2),
                error=f"Inference computation error: {str(e)}"
            )


# Global singleton instance
predictor = ASLPredictor()
