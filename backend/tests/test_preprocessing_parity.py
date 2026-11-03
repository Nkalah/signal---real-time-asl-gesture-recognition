import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from backend.app.config import SUPPORTED_LETTERS as BACKEND_LETTERS
from training.prepare_data import (
    LETTER_TO_INDEX,
    SUPPORTED_LETTERS as TRAINING_LETTERS,
    load_collected_dataset,
    load_collected_data,
    load_training_dataset,
    generate_canonical_hand,
    normalize_landmarks,
    split_dataset,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
COORDINATE_COLUMNS = [f"{axis}{i}" for i in range(21) for axis in "xyz"]
CSV_COLUMNS = [
    "label",
    "source",
    "participant_id",
    "session_id",
    "condition",
    "handedness",
    "captured_at",
    *COORDINATE_COLUMNS,
]


def _capture_row(letter, raw, session="session_one", handedness="Right", timestamp="2026-11-02T00:00:00Z"):
    row = {
        "label": letter,
        "source": "webcam",
        "participant_id": "signer_one",
        "session_id": session,
        "condition": "front_view",
        "handedness": handedness,
        "captured_at": timestamp,
    }
    row.update({
        column: float(value)
        for column, value in zip(COORDINATE_COLUMNS, np.asarray(raw).reshape(-1))
    })
    return row


def _write_capture_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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


def test_synthetic_template_csv_loader_smoke_normalizes_labels_and_groups(tmp_path):
    rows = [
        _capture_row(letter, generate_canonical_hand(letter))
        for letter in TRAINING_LETTERS
    ]
    _write_capture_csv(tmp_path / "signal_landmarks_signer_one.csv", rows)

    X, y, groups = load_collected_dataset(tmp_path)

    assert X.shape == (24, 63)
    assert y.tolist() == list(range(24))
    assert groups == ["session::signer_one::session_one"] * 24


def test_csv_loader_requires_all_static_letters_after_scale_filtering(tmp_path):
    _write_capture_csv(
        tmp_path / "signal_landmarks_signer_one.csv",
        [_capture_row("A", generate_canonical_hand("A"))],
    )

    try:
        load_collected_data(tmp_path)
    except ValueError as error:
        assert "missing classes" in str(error)
    else:
        raise AssertionError("A partial real dataset must not silently train.")


def test_training_loader_requires_real_data_unless_smoke_flag_is_set(tmp_path):
    try:
        load_training_dataset(data_dir=tmp_path)
    except FileNotFoundError as error:
        assert "--allow-synthetic" in str(error)
    else:
        raise AssertionError("Default training must fail without browser-collected data.")

    _, _, _, source = load_training_dataset(
        samples_per_class=1,
        data_dir=tmp_path,
        allow_synthetic=True,
    )
    assert source == "synthetic"


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


def test_real_csv_loader_concatenates_files_drops_small_hands_and_supports_none(tmp_path):
    letters = [
        _capture_row(
            letter,
            generate_canonical_hand(letter),
            session="session_one",
            handedness="Left" if index % 2 else "Right",
            timestamp=f"2026-11-02T00:{index:02d}:00Z",
        )
        for index, letter in enumerate(TRAINING_LETTERS)
    ]
    small_a = generate_canonical_hand("A") * 0.05
    none_row = _capture_row(
        "NONE",
        generate_canonical_hand("A"),
        timestamp="2026-11-02T00:25:00Z",
    )
    _write_capture_csv(tmp_path / "signal_landmarks_part_a.csv", letters[:12])
    _write_capture_csv(
        tmp_path / "signal_landmarks_part_b.csv",
        [_capture_row("A", small_a), *letters[12:], none_row],
    )

    data = load_collected_data(tmp_path)

    assert data is not None
    assert data.features.shape == (25, 63)
    assert data.label_names == [*TRAINING_LETTERS, "NONE"]
    assert data.dropped_small_hands == 1
    assert set(data.source_files) == {
        "signal_landmarks_part_a.csv",
        "signal_landmarks_part_b.csv",
    }
    left_index = data.handedness.index("Left")
    left_raw = generate_canonical_hand(TRAINING_LETTERS[1])
    np.testing.assert_allclose(
        data.features[left_index],
        normalize_landmarks(left_raw, "Left"),
        atol=1e-6,
    )


def test_browser_csv_loader_normalizes_raw_landmarks_and_splits_by_session(tmp_path):
    csv_path = tmp_path / "signal_landmarks_participant_01_sessions.csv"
    rows = [
        _capture_row(
            letter,
            generate_canonical_hand(letter),
            session=session_id,
            timestamp=f"2026-11-0{session_index + 1}T00:00:00Z",
        )
        for session_index, session_id in enumerate(("session_1", "session_2", "session_3"))
        for letter in TRAINING_LETTERS
    ]
    _write_capture_csv(csv_path, rows)

    X, y, groups = load_collected_dataset(tmp_path)
    assert X.shape == (72, 63)
    assert len(set(y.tolist())) == 24
    assert len(set(groups)) == 3
    assert set(groups) == {
        "session::signer_one::session_1",
        "session::signer_one::session_2",
        "session::signer_one::session_3",
    }

    train_idx, val_idx, test_idx, strategy = split_dataset(y, groups, seed=7)
    assert "same signer" in strategy
    train_groups = {groups[index] for index in train_idx}
    val_groups = {groups[index] for index in val_idx}
    test_groups = {groups[index] for index in test_idx}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)
