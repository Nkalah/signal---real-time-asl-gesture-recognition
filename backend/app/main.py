"""FastAPI main application for SIGNAL ASL Gesture Recognition."""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure app package is importable
app_dir = Path(__file__).resolve().parent
if str(app_dir.parent) not in sys.path:
    sys.path.insert(0, str(app_dir.parent))

from app.config import (
    CORS_ORIGINS,
    SUPPORTED_LETTERS,
    EXCLUDED_LETTERS,
    EXCLUSION_REASON,
    PREPROCESSING_VERSION,
    INPUT_FEATURES,
    NUM_CLASSES
)
from app.schemas import (
    HealthResponse,
    ReadyResponse,
    ModelMetadataResponse,
    PreprocessingConfig,
    LandmarkPredictionRequest,
    PredictionResponse
)
from app.predictor import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure model loaded
    predictor.load_model()
    yield
    # Shutdown cleanup if needed


app = FastAPI(
    title="SIGNAL ASL Gesture Recognition API",
    description="Real-time inference API for American Sign Language static alphabet landmark classification.",
    version="1.0.0",
    lifespan=lifespan
)

# Explicit CORS configuration for frontend dev & demo origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def get_health():
    """Returns basic service health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="SIGNAL ASL Gesture Recognition Service"
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Diagnostics"])
def get_ready():
    """Reports whether the PyTorch model artifact is loaded and ready for live inference."""
    if not predictor.is_ready:
        return ReadyResponse(
            ready=False,
            model_loaded=False,
            classes_count=NUM_CLASSES,
            device=str(predictor.device),
            message=predictor.load_error or "Model weights not found. Please train or provide model artifact."
        )
        
    return ReadyResponse(
        ready=True,
        model_loaded=True,
        classes_count=len(predictor.labels),
        device=str(predictor.device),
        message="Model is loaded and ready for real-time inference."
    )


@app.get("/metadata", response_model=ModelMetadataResponse, tags=["Model Info"])
def get_metadata():
    """Returns supported alphabet classes, feature dimensionalities, and preprocessing specifications."""
    return ModelMetadataResponse(
        model_name="SIGNAL ASL Landmark MLP",
        architecture=(
            predictor.model.get_config()["architecture"]
            if predictor.model is not None
            else f"MLP-63-128-64-{NUM_CLASSES}"
        ),
        classes_count=len(predictor.labels) if predictor.is_ready else NUM_CLASSES,
        supported_letters=SUPPORTED_LETTERS,
        excluded_letters=EXCLUDED_LETTERS,
        exclusion_reason=EXCLUSION_REASON,
        preprocessing=PreprocessingConfig(
            version=PREPROCESSING_VERSION,
            input_features=INPUT_FEATURES,
            reference_point="wrist (landmark 0)",
            normalization="wrist-relative; divide by XY distance from landmark 0 to landmark 9; mirror Left x",
            epsilon=1e-6
        ),
        input_dimension=INPUT_FEATURES,
        version="1.0.0",
        training_data_source=predictor.training_data_source,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_gesture(request: LandmarkPredictionRequest):
    """
    Classifies a 63-dimensional normalized hand landmark vector.
    
    Returns letter classification with softmax confidence and top predictions.
    If the model is not loaded, returns 503 SERVICE_UNAVAILABLE.
    """
    if not predictor.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=predictor.load_error or "Model is not loaded. Run the training workflow to generate model artifact."
        )
        
    result = predictor.predict(request.landmarks)
    if result.error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error
        )
        
    return result
