"""Data preparation and landmark preprocessing for SIGNAL ASL Gesture Recognition."""

import math
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from backend.app.preprocessing import normalize_landmarks

# 24 Static ASL Letters (Excluding J and Z)
SUPPORTED_LETTERS: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]
LETTER_TO_INDEX: Dict[str, int] = {letter: idx for idx, letter in enumerate(SUPPORTED_LETTERS)}
INDEX_TO_LETTER: Dict[int, str] = {idx: letter for idx, letter in enumerate(SUPPORTED_LETTERS)}
NONE_LABEL = "NONE"
CLASS_LABELS: List[str] = [*SUPPORTED_LETTERS, NONE_LABEL]
LETTER_TO_CLASS_INDEX: Dict[str, int] = {
    label: idx for idx, label in enumerate(CLASS_LABELS)
}
MIN_HAND_SCALE = 0.06
LOGGER = logging.getLogger(__name__)


@dataclass
class CollectedLandmarkData:
    features: np.ndarray
    labels: np.ndarray
    label_names: List[str]
    participants: List[str]
    sessions: List[str]
    conditions: List[str]
    handedness: List[str]
    timestamps_ns: np.ndarray
    timestamp_strings: List[str]
    source_files: List[str]
    dropped_small_hands: int


@dataclass
class EvaluationSplit:
    name: str
    method: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def load_collected_data(
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw",
    min_hand_scale: float = MIN_HAND_SCALE,
) -> CollectedLandmarkData | None:
    """Load and normalize every browser-collected CSV in data_dir."""
    csv_files = sorted(data_dir.glob("signal_landmarks_*.csv"))
    legacy_files = sorted(data_dir.glob("landmarks_*.npy"))
    if not csv_files:
        if legacy_files:
            raise ValueError(
                "Only legacy Python MediaPipe captures were found. "
                "Training requires browser-collected signal_landmarks_*.csv files."
            )
        return None
    if legacy_files:
        LOGGER.warning(
            "Ignoring %d legacy .npy files because they were captured with a different detector.",
            len(legacy_files),
        )

    coordinate_columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    required = {
        "label",
        "source",
        "participant_id",
        "session_id",
        "condition",
        "handedness",
        "captured_at",
        *coordinate_columns,
    }
    features: List[np.ndarray] = []
    labels: List[int] = []
    participants: List[str] = []
    sessions: List[str] = []
    conditions: List[str] = []
    handedness_values: List[str] = []
    timestamps_ns: List[int] = []
    timestamp_strings: List[str] = []
    source_files: List[str] = []
    seen_letters = set()
    dropped_small_hands = 0
    sample_count = 0

    for path in csv_files:
        df = pd.read_csv(path)
        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(f"CSV {path.name} is missing columns: {sorted(missing_columns)}")
        for row_number, row in enumerate(df.to_dict(orient="records"), start=2):
            sample_count += 1
            letter = str(row["label"]).strip().upper()
            if letter not in LETTER_TO_CLASS_INDEX:
                raise ValueError(f"Unsupported label {letter!r} in {path.name}:{row_number}")
            participant = str(row["participant_id"]).strip()
            session = str(row["session_id"]).strip()
            condition = str(row["condition"]).strip()
            if not participant or not session or not condition:
                raise ValueError(f"Missing collection metadata in {path.name}:{row_number}")
            hand = str(row["handedness"]).strip().title()
            if hand not in {"Left", "Right"}:
                raise ValueError(
                    f"Expected Left or Right handedness in {path.name}:{row_number}, got {hand!r}"
                )
            try:
                raw = np.asarray(
                    [row[column] for column in coordinate_columns], dtype=np.float32
                ).reshape((21, 3))
                timestamp = pd.to_datetime(row["captured_at"], utc=True, errors="raise")
            except (TypeError, ValueError, pd.errors.ParserError) as error:
                raise ValueError(f"Invalid capture row in {path.name}:{row_number}: {error}") from error
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"Non-finite landmarks in {path.name}:{row_number}")

            relative = raw - raw[0]
            scale = float(np.linalg.norm(relative[9, :2]))
            if scale < min_hand_scale:
                dropped_small_hands += 1
                continue

            features.append(normalize_landmarks(raw, hand))
            labels.append(LETTER_TO_CLASS_INDEX[letter])
            participants.append(participant)
            sessions.append(session)
            conditions.append(condition)
            handedness_values.append(hand)
            timestamps_ns.append(int(timestamp.value))
            timestamp_strings.append(timestamp.isoformat())
            source_files.append(path.name)
            seen_letters.add(letter)

    LOGGER.info(
        "Loaded %d browser capture rows from %d CSV files; dropped %d rows below scale %.2f.",
        sample_count,
        len(csv_files),
        dropped_small_hands,
        min_hand_scale,
    )
    missing_letters = sorted(set(SUPPORTED_LETTERS) - seen_letters)
    if missing_letters:
        raise ValueError(
            "Collected dataset is missing classes after hand-scale filtering: "
            + ", ".join(missing_letters)
        )
    if not features:
        raise ValueError("No usable landmark rows remain after hand-scale filtering.")

    label_names = [*SUPPORTED_LETTERS]
    if NONE_LABEL in seen_letters:
        label_names.append(NONE_LABEL)
    LOGGER.info("Training labels: %s", label_names)
    return CollectedLandmarkData(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        label_names=label_names,
        participants=participants,
        sessions=sessions,
        conditions=conditions,
        handedness=handedness_values,
        timestamps_ns=np.asarray(timestamps_ns, dtype=np.int64),
        timestamp_strings=timestamp_strings,
        source_files=source_files,
        dropped_small_hands=dropped_small_hands,
    )


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
    """Compatibility wrapper returning features, targets, and participant groups."""
    data = load_collected_data(data_dir)
    if data is None:
        return None

    groups = [
        f"session::{participant}::{session}"
        for participant, session in zip(data.participants, data.sessions)
    ]
    return data.features, data.labels, groups


def load_training_dataset(
    samples_per_class: int = 250,
    seed: int = 42,
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "raw",
    allow_synthetic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
    """Load real browser data unless generated smoke-test data is explicitly allowed."""
    collected = load_collected_dataset(data_dir)
    if collected is not None:
        return (*collected, "collected")

    if not allow_synthetic:
        raise FileNotFoundError(
            f"No browser-collected signal_landmarks_*.csv files found in {data_dir}. "
            "Collect real data before training, or pass --allow-synthetic for a smoke test."
        )

    X, y, participants = build_dataset(samples_per_class=samples_per_class, seed=seed)
    return X, y, participants, "synthetic"


def split_dataset(
    y: np.ndarray,
    participants: List[str],
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Make a deterministic sample split for generated smoke-test data only."""
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


def make_evaluation_splits(
    data: CollectedLandmarkData,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
    buffer_frames: int = 30,
) -> List[EvaluationSplit]:
    """Create leave-one-group-out folds or a buffered chronological same-session holdout."""
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("Validation and test fractions must be between 0 and 1.")
    if buffer_frames < 0:
        raise ValueError("buffer_frames must be non-negative.")

    participant_ids = sorted(set(data.participants))
    session_keys = sorted(set(zip(data.participants, data.sessions)))
    folds: List[EvaluationSplit] = []

    if len(participant_ids) >= 2:
        groups = [np.asarray(data.participants, dtype=object)]
        group_names = participant_ids
        method = "leave-one-participant-out"
    elif len(session_keys) >= 2:
        group_values = [
            f"{person}::{session}"
            for person, session in zip(data.participants, data.sessions)
        ]
        groups = [np.asarray(group_values, dtype=object)]
        group_names = sorted(set(group_values))
        method = "leave-one-session-out"
    else:
        per_class: Dict[str, List[np.ndarray]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for label_index, label in enumerate(data.label_names):
            indices = np.flatnonzero(data.labels == label_index)
            ordered = indices[
                np.argsort(data.timestamps_ns[indices], kind="stable")
            ]
            test_count = max(1, math.ceil(len(ordered) * test_fraction))
            test_start = len(ordered) - test_count
            buffer_start = test_start - buffer_frames
            validation_count = max(1, math.ceil(len(ordered) * validation_fraction))
            validation_start = buffer_start - validation_count
            if validation_start <= 0:
                raise ValueError(
                    f"Not enough time-ordered samples for {label} to create train, validation, "
                    f"a {buffer_frames}-frame buffer, and held-out test blocks."
                )
            per_class["train"].append(ordered[:validation_start])
            per_class["validation"].append(ordered[validation_start:buffer_start])
            per_class["test"].append(ordered[test_start:])

        folds.append(
            EvaluationSplit(
                name="single-recording",
                method="same-person, same-session, NOT generalization",
                train_indices=np.concatenate(per_class["train"]),
                validation_indices=np.concatenate(per_class["validation"]),
                test_indices=np.concatenate(per_class["test"]),
            )
        )
        return folds

    group_values = groups[0]
    for held_out in group_names:
        is_test_group = group_values == held_out
        test_indices = np.flatnonzero(is_test_group)
        training_pool = np.flatnonzero(~is_test_group)
        if not len(test_indices) or not len(training_pool):
            raise ValueError(f"Cannot create held-out fold for group {held_out!r}.")

        per_class_train: List[np.ndarray] = []
        per_class_validation: List[np.ndarray] = []
        for label_index, label in enumerate(data.label_names):
            test_class = test_indices[data.labels[test_indices] == label_index]
            pool_class = training_pool[data.labels[training_pool] == label_index]
            if not len(test_class):
                raise ValueError(f"Held-out group {held_out!r} has no samples for {label}.")
            ordered = pool_class[
                np.argsort(data.timestamps_ns[pool_class], kind="stable")
            ]
            validation_count = max(1, math.ceil(len(ordered) * validation_fraction))
            if len(ordered) <= validation_count:
                raise ValueError(
                    f"Training groups outside {held_out!r} do not have enough {label} "
                    "samples for both training and validation."
                )
            per_class_train.append(ordered[:-validation_count])
            per_class_validation.append(ordered[-validation_count:])

        folds.append(
            EvaluationSplit(
                name=str(held_out),
                method=method,
                train_indices=np.concatenate(per_class_train),
                validation_indices=np.concatenate(per_class_validation),
                test_indices=test_indices,
            )
        )
    return folds


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
