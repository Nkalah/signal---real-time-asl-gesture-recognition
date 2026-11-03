"""Pydantic schemas for the SIGNAL ASL Gesture Recognition API."""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field, field_validator
import math


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status", example="healthy")
    version: str = Field(..., description="API version", example="1.0.0")
    service: str = Field(default="SIGNAL ASL Gesture Recognition Service")


class PreprocessingConfig(BaseModel):
    version: str = Field(..., example="v2.0.0")
    input_features: int = Field(default=63, description="21 landmarks x 3 coordinates (x, y, z)")
    reference_point: str = Field(default="wrist (landmark 0)")
    normalization: str = Field(
        default="wrist-relative; divide by XY distance from landmark 0 to landmark 9; mirror Left x"
    )
    epsilon: float = Field(default=1e-6)


class ModelMetadataResponse(BaseModel):
    model_name: str
    architecture: str
    classes_count: int
    supported_letters: List[str]
    excluded_letters: List[str]
    exclusion_reason: str
    preprocessing: PreprocessingConfig
    input_dimension: int = 63
    version: str
    training_data_source: Optional[str] = None


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    classes_count: int
    device: str
    message: str


class LandmarkPredictionRequest(BaseModel):
    landmarks: List[float] = Field(
        ...,
        description="Exactly 63 normalized landmark coordinates [x0, y0, z0, ..., x20, y20, z20]",
        example=[0.0] * 63
    )

    @field_validator("landmarks")
    @classmethod
    def validate_landmarks_count_and_values(cls, v: List[float]) -> List[float]:
        if len(v) != 63:
            raise ValueError(f"Expected exactly 63 normalized coordinates (21 landmarks x 3), received {len(v)}")
        for idx, val in enumerate(v):
            if val is None or math.isnan(val) or math.isinf(val):
                raise ValueError(f"Landmark at index {idx} contains non-finite value: {val}")
            if abs(val) > 10.0:
                raise ValueError(f"Normalized landmark at index {idx} exceeds reasonable range: {val}")
        return v


class TopPrediction(BaseModel):
    letter: str
    confidence: float


class PredictionResponse(BaseModel):
    ready: bool
    letter: Optional[str] = None
    confidence: float = 0.0
    probabilities: Optional[Dict[str, float]] = None
    top_predictions: Optional[List[TopPrediction]] = None
    model_version: str
    inference_time_ms: float
    model_forward_time_ms: float = 0.0
    error: Optional[str] = None
