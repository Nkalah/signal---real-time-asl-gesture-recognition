"""Data preparation and landmark preprocessing for SIGNAL ASL Gesture Recognition."""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import GroupShuffleSplit, train_test_split

# 24 Static ASL Letters (Excluding J and Z)
SUPPORTED_LETTERS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]
LETTER_TO_INDEX: Dict[str, int] = {letter: idx for idx, letter in enumerate(SUPPORTED_LETTERS)}
INDEX_TO_LETTER: Dict[int, str] = {idx: letter for idx, letter in enumerate(SUPPORTED_LETTERS)}


def normalize_landmarks(coords: np.ndarray) -> np.ndarray:
    """
    Standard preprocessing specification v1.0.0:
    
    1. Input coords: shape (21, 3) or flattened (63,)
    2. Missing / NaN check: all coordinates must be finite.
    3. Wrist-relative transformation: p_i' = p_i - p_0 (where p_0 is Wrist, landmark 0).
    4. Hand-scale normalization:
       scale s = max Euclidean distance from wrist to any landmark point.
       p_i'' = (p_i - p_0) / (s + epsilon), where epsilon = 1e-6.
    5. Returns flattened 63-element float32 array: [x0, y0, z0, ..., x20, y20, z20].
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim == 1:
        if coords.shape[0] != 63:
            raise ValueError(f"Expected 63 coordinates, got {coords.shape[0]}")
        coords = coords.reshape((21, 3))
    elif coords.shape != (21, 3):
        raise ValueError(f"Expected shape (21, 3), got {coords.shape}")

    if not np.all(np.isfinite(coords)):
        raise ValueError("Landmarks contain non-finite values (NaN or Inf)")

    wrist = coords[0].copy()
    shifted = coords - wrist
    
    # Calculate Euclidean distances from wrist
    dists = np.linalg.norm(shifted, axis=1)
    scale = np.max(dists)
    
    epsilon = 1e-6
    if scale < epsilon:
        raise ValueError(f"Degenerate hand scale detected: {scale:.8f}")

    normalized = shifted / (scale + epsilon)
    return normalized.flatten().astype(np.float32)


def generate_canonical_hand(letter: str) -> np.ndarray:
    """
    Generates anatomically accurate 21 3D landmarks for a standard hand configuration
    signing the specified ASL letter.
    
    Landmark Indices:
      0: Wrist
      1-4: Thumb (CMC, MCP, IP, Tip)
      5-8: Index (MCP, PIP, DIP, Tip)
      9-12: Middle (MCP, PIP, DIP, Tip)
      13-16: Ring (MCP, PIP, DIP, Tip)
      17-20: Pinky (MCP, PIP, DIP, Tip)
    """
    pts = np.zeros((21, 3), dtype=np.float32)
    # Wrist
    pts[0] = [0.0, 0.0, 0.0]

    # Knuckle base positions (MCPs) relative to wrist
    pts[1] = [-0.25, 0.20, 0.05]   # Thumb CMC
    pts[5] = [-0.15, 0.55, 0.00]   # Index MCP
    pts[9] = [ 0.00, 0.58, 0.00]   # Middle MCP
    pts[13] = [ 0.14, 0.54, 0.00]  # Ring MCP
    pts[17] = [ 0.26, 0.48, 0.00]  # Pinky MCP

    # Helper function to place 3 segments (PIP, DIP, Tip) for a finger
    def place_finger(mcp_idx: int, length: float, direction: np.ndarray, curl: float = 0.0):
        # curl: 0.0 = straight, 1.0 = curled into palm
        mcp = pts[mcp_idx]
        seg = length / 3.0
        dir_norm = direction / np.linalg.norm(direction)
        
        # When curled, fold segments towards palm (-Z and down)
        fold_dir = np.array([0.0, -0.6, -0.8], dtype=np.float32)
        fold_dir = fold_dir / np.linalg.norm(fold_dir)
        
        eff_dir1 = (1.0 - curl * 0.4) * dir_norm + (curl * 0.4) * fold_dir
        eff_dir1 /= np.linalg.norm(eff_dir1)
        pts[mcp_idx + 1] = mcp + eff_dir1 * seg
        
        eff_dir2 = (1.0 - curl * 0.7) * dir_norm + (curl * 0.7) * fold_dir
        eff_dir2 /= np.linalg.norm(eff_dir2)
        pts[mcp_idx + 2] = pts[mcp_idx + 1] + eff_dir2 * seg
        
        eff_dir3 = (1.0 - curl) * dir_norm + curl * fold_dir
        eff_dir3 /= np.linalg.norm(eff_dir3)
        pts[mcp_idx + 3] = pts[mcp_idx + 2] + eff_dir3 * seg

    # Helper to place thumb
    def place_thumb(dir_vec: np.ndarray, curl: float = 0.0):
        cmc = pts[1]
        seg = 0.16
        pts[2] = cmc + dir_vec * seg
        pts[3] = pts[2] + dir_vec * seg * (1.0 - curl * 0.3) + np.array([curl * 0.1, 0, -curl * 0.1])
        pts[4] = pts[3] + dir_vec * seg * (1.0 - curl * 0.5) + np.array([curl * 0.2, 0, -curl * 0.2])

    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([0.0, 0.2, 0.9])
    fwd_horiz = np.array([-0.8, 0.3, 0.3])
    down = np.array([0.0, -0.8, 0.4])

    if letter == 'A':
        # Fist, thumb upright on side of index
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.1, 0.9, 0.1]), curl=0.1)

    elif letter == 'B':
        # Flat 4 fingers up, thumb folded across palm
        place_finger(5, 0.46, up, curl=0.0)
        place_finger(9, 0.50, up, curl=0.0)
        place_finger(13, 0.46, up, curl=0.0)
        place_finger(17, 0.38, up, curl=0.0)
        place_thumb(np.array([0.8, 0.2, -0.3]), curl=0.7)

    elif letter == 'C':
        # Curved arc
        c_curve = np.array([-0.2, 0.6, 0.6])
        place_finger(5, 0.45, c_curve, curl=0.45)
        place_finger(9, 0.48, c_curve, curl=0.45)
        place_finger(13, 0.45, c_curve, curl=0.45)
        place_finger(17, 0.38, c_curve, curl=0.45)
        place_thumb(np.array([0.2, 0.5, 0.6]), curl=0.4)

    elif letter == 'D':
        # Index straight up, others curled in circle touching thumb
        place_finger(5, 0.47, up, curl=0.0)
        place_finger(9, 0.48, up, curl=0.85)
        place_finger(13, 0.45, up, curl=0.85)
        place_finger(17, 0.38, up, curl=0.85)
        place_thumb(np.array([0.4, 0.5, 0.4]), curl=0.5)

    elif letter == 'E':
        # All fingers tightly clawed/curled down touching thumb below
        place_finger(5, 0.45, up, curl=0.98)
        place_finger(9, 0.48, up, curl=0.98)
        place_finger(13, 0.45, up, curl=0.98)
        place_finger(17, 0.38, up, curl=0.98)
        place_thumb(np.array([0.6, 0.1, -0.3]), curl=0.9)

    elif letter == 'F':
        # Index & thumb touching circle, 3 fingers extended
        place_finger(5, 0.45, np.array([0.1, 0.5, 0.5]), curl=0.8)
        place_finger(9, 0.50, up, curl=0.0)
        place_finger(13, 0.46, up, curl=0.0)
        place_finger(17, 0.38, up, curl=0.0)
        place_thumb(np.array([0.3, 0.6, 0.5]), curl=0.5)

    elif letter == 'G':
        # Index pointing forward/left horizontally, thumb parallel
        place_finger(5, 0.46, np.array([-0.9, 0.1, 0.3]), curl=0.0)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([-0.8, 0.2, 0.2]), curl=0.1)

    elif letter == 'H':
        # Index & middle extended horizontally together
        place_finger(5, 0.46, np.array([-0.9, 0.1, 0.2]), curl=0.0)
        place_finger(9, 0.48, np.array([-0.9, 0.05, 0.2]), curl=0.0)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.1, 0.5, -0.2]), curl=0.7)

    elif letter == 'I':
        # Pinky up, others curled in fist
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.40, up, curl=0.0)
        place_thumb(np.array([0.6, 0.3, -0.2]), curl=0.7)

    elif letter == 'K':
        # Index up, middle forward-up, thumb between them
        place_finger(5, 0.48, up, curl=0.0)
        place_finger(9, 0.48, np.array([0.1, 0.7, 0.6]), curl=0.05)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.2, 0.7, 0.4]), curl=0.2)

    elif letter == 'L':
        # Index up, thumb out at 90 degrees
        place_finger(5, 0.48, up, curl=0.0)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([-0.95, 0.1, 0.0]), curl=0.0)

    elif letter == 'M':
        # Fist with thumb under 3 fingers
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.45, 0.4, -0.4]), curl=0.85)

    elif letter == 'N':
        # Fist with thumb under 2 fingers
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.3, 0.4, -0.4]), curl=0.85)

    elif letter == 'O':
        # All fingers touching thumb in circle
        o_vec = np.array([0.1, 0.6, 0.6])
        place_finger(5, 0.45, o_vec, curl=0.65)
        place_finger(9, 0.48, o_vec, curl=0.65)
        place_finger(13, 0.45, o_vec, curl=0.65)
        place_finger(17, 0.38, o_vec, curl=0.65)
        place_thumb(np.array([0.3, 0.6, 0.55]), curl=0.55)

    elif letter == 'P':
        # 'K' pointing downward
        place_finger(5, 0.48, np.array([-0.8, -0.2, 0.4]), curl=0.0)
        place_finger(9, 0.48, np.array([0.0, -0.9, 0.3]), curl=0.0)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([-0.5, -0.4, 0.4]), curl=0.2)

    elif letter == 'Q':
        # 'G' pointing downward
        place_finger(5, 0.46, np.array([0.0, -0.95, 0.2]), curl=0.0)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([-0.2, -0.9, 0.1]), curl=0.1)

    elif letter == 'R':
        # Index and middle crossed
        place_finger(5, 0.48, np.array([0.1, 0.95, 0.1]), curl=0.0)
        place_finger(9, 0.49, np.array([-0.1, 0.95, 0.05]), curl=0.0)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.5, 0.3, -0.2]), curl=0.7)

    elif letter == 'S':
        # Tight fist with thumb over fingers
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.3, 0.4, 0.4]), curl=0.8)

    elif letter == 'T':
        # Thumb between index and middle
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.15, 0.45, 0.1]), curl=0.6)

    elif letter == 'U':
        # Index & middle together up
        place_finger(5, 0.48, up, curl=0.0)
        place_finger(9, 0.49, up, curl=0.0)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.5, 0.3, -0.2]), curl=0.7)

    elif letter == 'V':
        # Index & middle spread in 'V'
        place_finger(5, 0.48, np.array([-0.25, 0.95, 0.0]), curl=0.0)
        place_finger(9, 0.49, np.array([ 0.25, 0.95, 0.0]), curl=0.0)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.5, 0.3, -0.2]), curl=0.7)

    elif letter == 'W':
        # 3 fingers up in 'W'
        place_finger(5, 0.48, np.array([-0.3, 0.95, 0.0]), curl=0.0)
        place_finger(9, 0.50, up, curl=0.0)
        place_finger(13, 0.47, np.array([ 0.3, 0.95, 0.0]), curl=0.0)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.4, 0.2, -0.2]), curl=0.7)

    elif letter == 'X':
        # Index crooked/hooked
        place_finger(5, 0.46, up, curl=0.6)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.38, up, curl=0.95)
        place_thumb(np.array([0.5, 0.3, -0.2]), curl=0.7)

    elif letter == 'Y':
        # Thumb & pinky out, 3 middle curled
        place_finger(5, 0.45, up, curl=0.95)
        place_finger(9, 0.48, up, curl=0.95)
        place_finger(13, 0.45, up, curl=0.95)
        place_finger(17, 0.42, np.array([0.6, 0.7, 0.0]), curl=0.0)
        place_thumb(np.array([-0.8, 0.4, 0.0]), curl=0.0)

    return pts


def augment_landmarks(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Applies realistic 3D spatial augmentations:
    - 3D Euler rotation (-20 to +20 degrees)
    - Uniform scale jitter (0.85x to 1.15x)
    - Joint coordinate jitter
    """
    pts = landmarks.copy()
    
    # Random 3D rotation angles (radians)
    rx = rng.uniform(-0.35, 0.35)
    ry = rng.uniform(-0.35, 0.35)
    rz = rng.uniform(-0.35, 0.35)
    
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)]
    ])
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1]
    ])
    R = Rz @ Ry @ Rx
    
    # Rotate around wrist
    rotated = pts @ R.T
    
    # Scale variation
    scale = rng.uniform(0.85, 1.15)
    scaled = rotated * scale
    
    # Small joint position jitter
    jitter = rng.normal(0.0, 0.012, size=scaled.shape)
    jitter[0] = 0.0  # keep wrist stable
    augmented = scaled + jitter

    if rng.random() < 0.5:
        augmented[:, 0] *= -1.0

    return augmented


def load_collected_dataset(
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw",
) -> Tuple[np.ndarray, np.ndarray, List[str]] | None:
    """Load browser-exported raw-landmark CSVs and legacy NumPy captures.

    Browser CSVs contain raw MediaPipe Tasks Vision image landmarks plus
    participant/session metadata. They are normalized here using the exact
    preprocessing function used by training and inference.
    """
    npy_files = sorted(data_dir.glob("landmarks_*.npy"))
    csv_files = sorted(data_dir.glob("signal_landmarks_*.csv"))
    if not npy_files and not csv_files:
        return None

    features: List[np.ndarray] = []
    labels: List[int] = []
    participant_ids: List[str] = []
    session_ids: List[str] = []
    seen_letters = set()

    coordinate_columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]

    for path in csv_files:
        df = pd.read_csv(path)
        required = {"label", "participant_id", "session_id", *coordinate_columns}
        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV {path.name} is missing columns: {sorted(missing_columns)}")
        for row in df.to_dict(orient="records"):
            letter = str(row["label"]).strip().upper()
            if letter not in LETTER_TO_INDEX:
                raise ValueError(f"Unsupported letter '{letter}' in {path.name}")
            raw = np.asarray([row[column] for column in coordinate_columns], dtype=np.float32).reshape(21, 3)
            normalized = normalize_landmarks(raw)
            features.append(normalized)
            labels.append(LETTER_TO_INDEX[letter])
            participant_ids.append(str(row["participant_id"]).strip())
            session_ids.append(str(row["session_id"]).strip())
            seen_letters.add(letter)

    for path in npy_files:
        parts = path.stem.split("_")
        if len(parts) < 4 or parts[0] != "landmarks":
            raise ValueError(f"Unexpected collected-data filename: {path.name}")
        letter = parts[1].upper()
        if letter not in LETTER_TO_INDEX:
            raise ValueError(f"Unsupported letter in collected-data filename: {path.name}")
        participant = "_".join(parts[2:-1])
        samples = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
        if samples.ndim == 3 and samples.shape[1:] == (21, 3):
            samples = np.stack([normalize_landmarks(sample) for sample in samples])
        elif samples.ndim != 2 or samples.shape[1] != 63:
            raise ValueError(
                f"Expected normalized (N, 63) or raw (N, 21, 3) samples in {path.name}, "
                f"got {samples.shape}"
            )
        if samples.shape[0] == 0 or not np.all(np.isfinite(samples)):
            raise ValueError(f"Collected data is empty or non-finite: {path.name}")
        wrist_relative = samples.reshape((-1, 21, 3))
        hand_scales = np.linalg.norm(wrist_relative, axis=2).max(axis=1)
        if np.any(hand_scales < 1e-6):
            raise ValueError(f"Collected data contains a degenerate hand: {path.name}")
        features.extend(samples)
        labels.extend([LETTER_TO_INDEX[letter]] * len(samples))
        participant_ids.extend([participant] * len(samples))
        # Legacy .npy files have no session metadata; preserve their participant grouping.
        session_ids.extend([participant] * len(samples))
        seen_letters.add(letter)

    missing = sorted(set(SUPPORTED_LETTERS) - seen_letters)
    if missing:
        raise ValueError(
            "Collected dataset is missing classes: "
            + ", ".join(missing)
            + ". Collect all 24 supported letters before training."
        )

    # Prevent near-duplicate frames from one recording session leaking across splits.
    # With >=3 real signers, split by signer. With one signer and >=3 sessions,
    # split by session (same-signer generalization only, not signer-independent).
    unique_people = set(participant_ids)
    unique_sessions = {(person, session) for person, session in zip(participant_ids, session_ids)}
    if len(unique_people) >= 3:
        groups = [f"signer::{person}" for person in participant_ids]
    elif len(unique_sessions) >= 3:
        groups = [f"session::{person}::{session}" for person, session in zip(participant_ids, session_ids)]
    else:
        groups = participant_ids

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        groups,
    )


def load_training_dataset(
    samples_per_class: int = 250,
    seed: int = 42,
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw",
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Prefer captured landmarks; use generated poses only when no captures exist."""
    collected = load_collected_dataset(data_dir)
    if collected is not None:
        return (*collected, "collected")

    X, y, participants = build_dataset(samples_per_class=samples_per_class, seed=seed)
    return X, y, participants, "synthetic"


def split_dataset(
    y: np.ndarray,
    participants: List[str],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Return train, validation, and test indices without signer overlap when possible."""
    indices = np.arange(len(y))
    unique_participants = set(participants)

    if len(unique_participants) >= 3:
        groups = np.asarray(participants)
        test_split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        train_val_idx, test_idx = next(test_split.split(indices, y, groups))
        val_split = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed + 1)
        train_idx, val_idx = next(
            val_split.split(
                train_val_idx,
                y[train_val_idx],
                groups[train_val_idx],
            )
        )
        train_idx = train_val_idx[train_idx]
        val_idx = train_val_idx[val_idx]

        expected_classes = set(range(len(SUPPORTED_LETTERS)))
        for split_name, split_idx in (
            ("training", train_idx),
            ("validation", val_idx),
            ("test", test_idx),
        ):
            missing = expected_classes - set(y[split_idx].tolist())
            if missing:
                missing_letters = [SUPPORTED_LETTERS[index] for index in sorted(missing)]
                raise ValueError(
                    f"Signer-based {split_name} split is missing classes "
                    f"{missing_letters}. Ensure each signer contributes every letter."
                )
        group_values = list(unique_participants)
        if all(str(value).startswith("session::") for value in group_values):
            split_name = "session-independent (same signer; not signer-independent)"
        else:
            split_name = "signer-independent"
        return train_idx, val_idx, test_idx, split_name

    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=y
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.1765,
        random_state=seed,
        stratify=y[train_val_idx],
    )
    return train_idx, val_idx, test_idx, "sample-wise (signer overlap possible)"


def build_dataset(samples_per_class: int = 150, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates a full preprocessed dataset of 63-dimensional feature vectors and labels.
    """
    rng = np.random.default_rng(seed)
    X_list = []
    y_list = []
    participant_list = []
    
    for class_idx, letter in enumerate(SUPPORTED_LETTERS):
        base_hand = generate_canonical_hand(letter)
        
        for s in range(samples_per_class):
            # Simulate 5 distinct participants / sessions
            participant_id = f"participant_{s % 5 + 1}"
            
            aug_hand = augment_landmarks(base_hand, rng)
            features = normalize_landmarks(aug_hand)
            
            X_list.append(features)
            y_list.append(class_idx)
            participant_list.append(participant_id)
            
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, participant_list


if __name__ == "__main__":
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating ASL landmark dataset across 24 classes...")
    X, y, participants = build_dataset(samples_per_class=200, seed=42)
    
    df = pd.DataFrame(X)
    df["label"] = y
    df["letter"] = [SUPPORTED_LETTERS[idx] for idx in y]
    df["participant"] = participants
    
    train_path = out_dir / "asl_landmarks.csv"
    df.to_csv(train_path, index=False)
    print(f"Saved {len(df)} samples ({X.shape[1]} features) to {train_path}")
