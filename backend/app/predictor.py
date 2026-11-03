"""Predictor module for ASL landmark inference."""

import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F

from app.config import (
    MODEL_WEIGHTS_PATH,
    MODEL_CONFIG_PATH,
    SUPPORTED_LETTERS,
    INPUT_FEATURES,
    NUM_CLASSES,
    PREPROCESSING_VERSION
)
from app.model import ASLLandmarkMLP
from app.schemas import PredictionResponse, TopPrediction


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
                
            # Initialize architecture
            model = ASLLandmarkMLP(
                input_dim=INPUT_FEATURES,
                hidden_dim_1=128,
                hidden_dim_2=64,
                num_classes=NUM_CLASSES
            )
            
            # Load state dict
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            self.model = model
            self.is_ready = True
            self.load_error = None
            
            # Load config if present
            if self.config_path.exists():
                try:
                    with open(self.config_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        if "supported_letters" in cfg:
                            self.labels = cfg["supported_letters"]
                        if "version" in cfg:
                            self.model_version = cfg["version"]
                        self.training_data_source = cfg.get(
                            "training_metadata", {}
                        ).get("dataset_source", "unknown")
                except Exception:
                    pass
                    
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
                probabilities = F.softmax(logits, dim=1).squeeze(0)
                
            # Top prediction
            conf, pred_idx = torch.max(probabilities, dim=0)
            best_idx = int(pred_idx.item())
            best_letter = self.labels[best_idx]
            best_conf = float(conf.item())
            
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
