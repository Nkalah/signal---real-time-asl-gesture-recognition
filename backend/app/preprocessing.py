"""Canonical ASL landmark preprocessing shared by training and the backend."""

import numpy as np


PREPROCESSING_VERSION = "v2.0.0"
INPUT_FEATURES = 63
LANDMARK_COUNT = 21
EPSILON = 1e-6


def normalize_landmarks(
    coords: np.ndarray,
    handedness: str = "Right",
) -> np.ndarray:
    """Translate to the wrist, scale by wrist-to-middle-MCP XY distance, and mirror left hands."""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 1:
        if coords.shape[0] != INPUT_FEATURES:
            raise ValueError(f"Expected 63 coordinates, got {coords.shape[0]}")
        coords = coords.reshape((LANDMARK_COUNT, 3))
    elif coords.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected shape (21, 3), got {coords.shape}")

    if not np.all(np.isfinite(coords)):
        raise ValueError("Landmarks contain non-finite values (NaN or Inf)")

    normalized_handedness = handedness.strip().lower()
    if normalized_handedness not in {"left", "right"}:
        raise ValueError(f"Expected handedness 'Left' or 'Right', got {handedness!r}")

    shifted = coords - coords[0]
    scale = float(np.linalg.norm(shifted[9, :2]))
    if scale < EPSILON:
        raise ValueError(f"Degenerate wrist-to-middle-MCP scale detected: {scale:.8f}")

    normalized = shifted / (scale + EPSILON)
    if normalized_handedness == "left":
        normalized[:, 0] *= -1

    return normalized.reshape(INPUT_FEATURES).astype(np.float32)
