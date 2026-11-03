"""Configuration module for SIGNAL ASL Backend."""

import os
from pathlib import Path
from typing import List
from backend.app.preprocessing import PREPROCESSING_VERSION

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_WEIGHTS_PATH = MODELS_DIR / "asl_mlp_model.pth"
MODEL_CONFIG_PATH = MODELS_DIR / "config.json"

# Supported classes: 24 static letters
# J and Z require dynamic movement and are excluded from static pose classification
SUPPORTED_LETTERS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]
EXCLUDED_LETTERS: List[str] = ["J", "Z"]
EXCLUSION_REASON: str = "Letters J and Z require motion trajectories and are not reliably recognizable from static 3D landmark poses."

# Preprocessing specification
INPUT_FEATURES = 63  # 21 landmarks * 3 coordinates (x, y, z)
NUM_CLASSES = len(SUPPORTED_LETTERS)

# Inference defaults
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("SIGNAL_CONFIDENCE_THRESHOLD", "0.60"))
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]
