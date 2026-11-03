"""Reproducible real-data training pipeline for the SIGNAL landmark classifier."""

import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from backend.app.config import INPUT_FEATURES, PREPROCESSING_VERSION, SUPPORTED_LETTERS
from backend.app.model import ASLLandmarkMLP
from training.calibration import calibrate_validation
from training.prepare_data import (
    CollectedLandmarkData,
    EvaluationSplit,
    build_dataset,
    load_collected_data,
    make_evaluation_splits,
    split_dataset,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent.parent / "backend" / "models"


def augment_normalized_features(
    batch: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply mild in-plane and local-joint variation without flipping sign orientation."""
    points = np.asarray(batch, dtype=np.float32).reshape((-1, 21, 3)).copy()
    angles = rng.uniform(-np.deg2rad(15), np.deg2rad(15), size=len(points))
    scale_x = rng.uniform(0.9, 1.1, size=len(points))
    scale_y = rng.uniform(0.9, 1.1, size=len(points))

    x = points[:, :, 0].copy()
    y = points[:, :, 1].copy()
    cosines = np.cos(angles)[:, None]
    sines = np.sin(angles)[:, None]
    points[:, :, 0] = (x * cosines - y * sines) * scale_x[:, None]
    points[:, :, 1] = (x * sines + y * cosines) * scale_y[:, None]

    points += rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
    joint_mask = rng.random(size=(len(points), 21, 1)) < 0.2
    jitter = rng.normal(0.0, 0.01, size=points.shape).astype(np.float32)
    points += jitter * joint_mask
    points[:, 0, :] = 0.0
    return points.reshape((-1, INPUT_FEATURES))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def _class_weights(targets: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing classes: {np.flatnonzero(counts == 0).tolist()}")
    weights = len(targets) / (num_classes * counts)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _fit_fold(
    X: np.ndarray,
    y: np.ndarray,
    split: EvaluationSplit,
    labels: List[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int = 12,
) -> Dict[str, object]:
    _set_seed(seed)
    train_X = torch.from_numpy(X[split.train_indices].astype(np.float32))
    train_y = torch.from_numpy(y[split.train_indices].astype(np.int64))
    val_X = torch.from_numpy(X[split.validation_indices].astype(np.float32))
    val_y = torch.from_numpy(y[split.validation_indices].astype(np.int64))
    test_X = torch.from_numpy(X[split.test_indices].astype(np.float32))
    test_y = torch.from_numpy(y[split.test_indices].astype(np.int64))

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    rng = np.random.default_rng(seed)
    model = ASLLandmarkMLP(
        input_dim=INPUT_FEATURES,
        hidden_dim_1=128,
        hidden_dim_2=64,
        num_classes=len(labels),
        dropout_rate=0.25,
    )
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(y[split.train_indices], len(labels)),
        label_smoothing=0.05,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in loader:
            augmented = augment_normalized_features(batch_X.numpy(), rng)
            inputs = torch.from_numpy(augmented)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X)
            val_loss = float(criterion(val_logits, val_y).item())
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 10 == 0 or epochs_without_improvement >= patience:
            LOGGER.info(
                "Fold %s epoch %d/%d: validation loss %.5f",
                split.name,
                epoch,
                epochs,
                val_loss,
            )
        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint was produced for fold {split.name}.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_logits = model(val_X).numpy()
        test_logits = model(test_X).numpy()
    test_predictions = test_logits.argmax(axis=1)

    return {
        "name": split.name,
        "method": split.method,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "training_seconds": time.perf_counter() - started,
        "train_indices": split.train_indices,
        "validation_indices": split.validation_indices,
        "test_indices": split.test_indices,
        "validation_logits": validation_logits,
        "validation_targets": val_y.numpy(),
        "test_logits": test_logits,
        "test_targets": test_y.numpy(),
        "test_predictions": test_predictions,
        "test_accuracy": float(accuracy_score(test_y.numpy(), test_predictions)),
        "test_macro_f1": float(
            f1_score(test_y.numpy(), test_predictions, labels=list(range(len(labels))), average="macro", zero_division=0)
        ),
    }


def _metadata(data: CollectedLandmarkData) -> Dict[str, object]:
    counts = {
        label: int(np.count_nonzero(data.labels == index))
        for index, label in enumerate(data.label_names)
    }
    return {
        "dataset_source": "collected",
        "data_files": sorted(set(data.source_files)),
        "participants": sorted(set(data.participants)),
        "participant_count": len(set(data.participants)),
        "sessions": sorted(set(data.sessions)),
        "session_count": len(set(zip(data.participants, data.sessions))),
        "conditions": sorted(set(data.conditions)),
        "sample_counts_per_label": counts,
        "sample_counts_per_hand": {
            hand: data.handedness.count(hand)
            for hand in ("Left", "Right")
        },
        "dropped_small_hands": data.dropped_small_hands,
    }


def train_model(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    seed: int = 42,
    save_dir: Path = DEFAULT_SAVE_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    allow_synthetic: bool = False,
) -> Dict[str, object]:
    """Fit fold models from browser data and export the selected best checkpoint."""
    _set_seed(seed)
    data = load_collected_data(data_dir)
    if data is None:
        if not allow_synthetic:
            raise FileNotFoundError(
                f"No browser-collected signal_landmarks_*.csv files found in {data_dir}. "
                "Collect real data before training, or pass --allow-synthetic for a smoke test."
            )
        X, y, participants = build_dataset(samples_per_class=250, seed=seed)
        labels = list(SUPPORTED_LETTERS)
        train_idx, val_idx, test_idx, method = split_dataset(y, participants, seed)
        splits = [
            EvaluationSplit(
                name="synthetic-smoke",
                method=f"synthetic smoke test ({method})",
                train_indices=train_idx,
                validation_indices=val_idx,
                test_indices=test_idx,
            )
        ]
        metadata: Dict[str, object] = {
            "dataset_source": "synthetic",
            "data_files": [],
            "participants": [],
            "participant_count": 0,
            "sessions": [],
            "session_count": 0,
            "conditions": [],
            "sample_counts_per_label": {},
            "sample_counts_per_hand": {},
            "dropped_small_hands": 0,
        }
        LOGGER.warning("SMOKE TEST ONLY: generated canonical poses are not real-world data.")
        if save_dir == DEFAULT_SAVE_DIR:
            save_dir = DEFAULT_SAVE_DIR / "smoke"
    else:
        X, y, labels = data.features, data.labels, data.label_names
        splits = make_evaluation_splits(data)
        metadata = _metadata(data)

    LOGGER.info(
        "Training with seed=%d, epochs=%d, batch_size=%d, learning_rate=%g",
        seed,
        epochs,
        batch_size,
        lr,
    )
    LOGGER.info(
        "Rows=%d, labels=%s, folds=%d",
        len(X),
        labels,
        len(splits),
    )

    fold_results = [
        _fit_fold(
            X,
            y,
            split,
            labels,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            seed=seed + index,
        )
        for index, split in enumerate(splits)
    ]
    selected_fold = min(fold_results, key=lambda result: result["best_validation_loss"])
    calibration = calibrate_validation(
        selected_fold["validation_logits"],
        selected_fold["validation_targets"],
        labels,
    )
    y_true = np.concatenate([result["test_targets"] for result in fold_results])
    y_pred = np.concatenate([result["test_predictions"] for result in fold_results])
    logits = np.concatenate([result["test_logits"] for result in fold_results])
    test_accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, labels=list(range(len(labels))), average="macro", zero_division=0)
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / "asl_mlp_model.pth"
    torch.save(selected_fold["state_dict"], checkpoint_path)
    weights = {
        key: value.detach().cpu().numpy().tolist()
        for key, value in selected_fold["state_dict"].items()
    }
    (save_dir / "weights.json").write_text(
        json.dumps(weights, separators=(",", ":")),
        encoding="utf-8",
    )
    (save_dir / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n",
        encoding="utf-8",
    )
    (save_dir / "calibration.json").write_text(
        json.dumps(calibration, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        save_dir / "evaluation_test.npz",
        X=X[np.concatenate([result["test_indices"] for result in fold_results])],
        y=y_true,
        logits=logits,
        predictions=y_pred,
    )

    config = {
        "model_name": "SIGNAL ASL Landmark MLP",
        "version": "2.0.0",
        "architecture": f"MLP-63-128-64-{len(labels)}",
        "input_features": INPUT_FEATURES,
        "num_classes": len(labels),
        "labels": labels,
        "supported_letters": SUPPORTED_LETTERS,
        "excluded_letters": ["J", "Z"],
        "exclusion_reason": "Letters J and Z require dynamic motion trajectories.",
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "reference_point": "wrist (landmark 0)",
            "scale_normalization": "XY distance from wrist (landmark 0) to middle MCP (landmark 9) plus 1e-6",
            "left_hand_mirroring": "negate normalized x for Left handedness",
            "epsilon": 1e-6,
        },
        "training_metadata": {
            **metadata,
            "seed": seed,
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "dropout": 0.25,
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "augmentation": "normalized XY rotation +/-15 degrees, independent XY scales 0.9-1.1, gaussian noise sigma 0.01, landmark jitter sigma 0.01; no flips",
            "evaluation_split": splits[0].method if len({split.method for split in splits}) == 1 else "leave-one-group-out",
            "folds": [
                {
                    "name": result["name"],
                    "method": result["method"],
                    "best_epoch": result["best_epoch"],
                    "best_validation_loss": result["best_validation_loss"],
                    "test_accuracy": result["test_accuracy"],
                    "test_macro_f1": result["test_macro_f1"],
                }
                for result in fold_results
            ],
            "mean_fold_test_accuracy": float(np.mean([result["test_accuracy"] for result in fold_results])),
            "mean_fold_test_macro_f1": float(np.mean([result["test_macro_f1"] for result in fold_results])),
            "test_accuracy": test_accuracy,
            "test_macro_f1": macro_f1,
            "selected_checkpoint_fold": selected_fold["name"],
        },
        "calibration": calibration,
        "confusion_matrix": matrix.tolist(),
    }
    (save_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (save_dir / "metrics_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    LOGGER.info(
        "Held-out accuracy=%.6f macro-F1=%.6f across %d fold(s)",
        test_accuracy,
        macro_f1,
        len(fold_results),
    )
    LOGGER.info("Saved best checkpoint and labels to %s", save_dir)
    return {
        "accuracy": test_accuracy,
        "macro_f1": macro_f1,
        "folds": fold_results,
        "config": config,
        "checkpoint": checkpoint_path,
    }
