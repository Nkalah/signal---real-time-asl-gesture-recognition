import json
import subprocess
from pathlib import Path

import numpy as np

from backend.app.config import SUPPORTED_LETTERS as BACKEND_LETTERS
from training.prepare_data import (
    LETTER_TO_INDEX,
    SUPPORTED_LETTERS as TRAINING_LETTERS,
    load_collected_dataset,
    generate_canonical_hand,
    normalize_landmarks,
    split_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


def test_python_and_typescript_match_python_generated_golden_vectors():
    fixture_path = ROOT_DIR / "backend" / "tests" / "fixtures" / "preprocessing_v2_golden.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["preprocessing_version"] == "v2.0.0"

    script = (
        "import { preprocessLandmarks } from './src/lib/api.ts';"
        "const input = JSON.parse(process.argv[1]);"
        "console.log(JSON.stringify(preprocessLandmarks(input.landmarks, input.handedness)));"
    )
    for case in fixture["cases"]:
        raw = np.asarray(case["raw_landmarks"], dtype=np.float32)
        python_features = normalize_landmarks(raw, case["handedness"])
        np.testing.assert_allclose(python_features, case["expected"], rtol=0, atol=1e-7)

        landmarks = [
            {"x": float(x), "y": float(y), "z": float(z)}
            for x, y, z in raw
        ]
        completed = subprocess.run(
            [
                "node",
                "--import",
                "tsx/esm",
                "--input-type=module",
                "-e",
                script,
                json.dumps({"landmarks": landmarks, "handedness": case["handedness"]}),
            ],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        typescript_features = np.asarray(json.loads(completed.stdout), dtype=np.float32)
        assert typescript_features.shape == (63,)
        np.testing.assert_allclose(typescript_features, python_features, rtol=0, atol=1e-5)


def test_supported_letter_order_matches_training_and_backend():
    assert TRAINING_LETTERS == BACKEND_LETTERS
    assert len(TRAINING_LETTERS) == 24
    assert "J" not in TRAINING_LETTERS
    assert "Z" not in TRAINING_LETTERS
    assert [LETTER_TO_INDEX[letter] for letter in TRAINING_LETTERS] == list(range(24))
    model_config = json.loads(
        (ROOT_DIR / "backend" / "models" / "config.json").read_text(encoding="utf-8")
    )
    assert model_config["supported_letters"] == TRAINING_LETTERS


def test_collected_samples_are_loaded_with_labels_and_participant_groups(tmp_path):
    for letter in TRAINING_LETTERS:
        normalized = normalize_landmarks(generate_canonical_hand(letter))
        np.save(
            tmp_path / f"landmarks_{letter}_signer_one_123.npy",
            normalized[np.newaxis, :],
        )

    X, y, participants = load_collected_dataset(tmp_path)

    assert X.shape == (24, 63)
    assert y.tolist() == list(range(24))
    assert participants == ["signer_one"] * 24


def test_collected_dataset_requires_every_supported_letter(tmp_path):
    normalized = normalize_landmarks(generate_canonical_hand("A"))
    np.save(tmp_path / "landmarks_A_signer_one_123.npy", normalized[np.newaxis, :])

    try:
        load_collected_dataset(tmp_path)
    except ValueError as error:
        assert "missing classes" in str(error)
    else:
        raise AssertionError("A partial collected dataset must not silently train.")


def test_signer_split_keeps_participants_out_of_other_splits():
    labels = np.tile(np.arange(24, dtype=np.int64), 3)
    participants = [signer for signer in ("one", "two", "three") for _ in range(24)]

    train_idx, val_idx, test_idx, strategy = split_dataset(labels, participants, seed=7)

    assert strategy == "signer-independent"
    train_signers = {participants[index] for index in train_idx}
    val_signers = {participants[index] for index in val_idx}
    test_signers = {participants[index] for index in test_idx}
    assert train_signers.isdisjoint(val_signers)
    assert train_signers.isdisjoint(test_signers)
    assert val_signers.isdisjoint(test_signers)


def test_browser_csv_loader_normalizes_raw_landmarks_and_splits_by_session(tmp_path):
    import csv

    coordinate_columns = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]
    columns = ["label", "source", "participant_id", "session_id", "condition", "handedness", "captured_at", *coordinate_columns]
    csv_path = tmp_path / "signal_landmarks_participant_01_sessions.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for session_id in ("session_1", "session_2", "session_3"):
            for letter in TRAINING_LETTERS:
                row = {
                    "label": letter,
                    "source": "webcam",
                    "participant_id": "participant_01",
                    "session_id": session_id,
                    "condition": "front_view",
                    "handedness": "Right",
                    "captured_at": "2026-11-02T00:00:00Z",
                }
                raw = generate_canonical_hand(letter).reshape(-1)
                row.update({column: float(value) for column, value in zip(coordinate_columns, raw)})
                writer.writerow(row)

    X, y, groups = load_collected_dataset(tmp_path)
    assert X.shape == (72, 63)
    assert len(set(y.tolist())) == 24
    assert len(set(groups)) == 3
    assert all(group.startswith("session::participant_01::") for group in groups)

    train_idx, val_idx, test_idx, strategy = split_dataset(y, groups, seed=7)
    assert "same signer" in strategy
    train_groups = {groups[index] for index in train_idx}
    val_groups = {groups[index] for index in val_idx}
    test_groups = {groups[index] for index in test_idx}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
