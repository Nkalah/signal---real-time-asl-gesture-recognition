import numpy as np
import pytest

from training.prepare_data import CollectedLandmarkData, make_evaluation_splits


LABELS = ["A", "B"]


def _dataset(participant_sessions, samples_per_label):
    features = []
    labels = []
    participants = []
    sessions = []
    timestamps = []
    for participant, session, offset in participant_sessions:
        for label_index in range(len(LABELS)):
            for sample_index in range(samples_per_label):
                features.append(np.zeros(63, dtype=np.float32))
                labels.append(label_index)
                participants.append(participant)
                sessions.append(session)
                timestamps.append(offset + sample_index)
    return CollectedLandmarkData(
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        label_names=LABELS,
        participants=participants,
        sessions=sessions,
        conditions=["test"] * len(labels),
        handedness=["Right"] * len(labels),
        timestamps_ns=np.asarray(timestamps, dtype=np.int64),
        timestamp_strings=[""] * len(labels),
        source_files=["fixture.csv"] * len(labels),
        dropped_small_hands=0,
    )


def test_single_recording_uses_last_time_block_and_30_frame_buffer():
    data = _dataset([("participant_01", "session_01", 0)], samples_per_label=100)

    split = make_evaluation_splits(data)[0]

    assert split.method == "same-person, same-session, NOT generalization"
    for label_index in range(len(LABELS)):
        train = split.train_indices[data.labels[split.train_indices] == label_index]
        validation = split.validation_indices[
            data.labels[split.validation_indices] == label_index
        ]
        test = split.test_indices[data.labels[split.test_indices] == label_index]
        assert len(train) == 35
        assert len(validation) == 15
        assert len(test) == 20
        assert data.timestamps_ns[test].min() - data.timestamps_ns[validation].max() == 31
        assert data.timestamps_ns[test].min() - data.timestamps_ns[train].max() > 30


def test_multiple_participants_use_leave_one_participant_out_folds():
    data = _dataset(
        [(f"participant_{i}", "session_01", i * 1000) for i in range(3)],
        samples_per_label=40,
    )

    folds = make_evaluation_splits(data)

    assert len(folds) == 3
    assert {fold.method for fold in folds} == {"leave-one-participant-out"}
    for fold in folds:
        test_people = {data.participants[index] for index in fold.test_indices}
        assert test_people == {fold.name}
        assert not test_people.intersection(
            data.participants[index] for index in fold.train_indices
        )
        assert not test_people.intersection(
            data.participants[index] for index in fold.validation_indices
        )
        assert set(data.labels[fold.test_indices]) == {0, 1}


def test_multiple_sessions_use_leave_one_session_out_folds():
    data = _dataset(
        [("participant_01", "session_01", 0), ("participant_01", "session_02", 1000)],
        samples_per_label=40,
    )

    folds = make_evaluation_splits(data)

    assert len(folds) == 2
    assert {fold.method for fold in folds} == {"leave-one-session-out"}
    for fold in folds:
        test_sessions = {data.sessions[index] for index in fold.test_indices}
        assert len(test_sessions) == 1
        assert test_sessions == {fold.name.split("::", maxsplit=1)[1]}
        assert not test_sessions.intersection(
            data.sessions[index] for index in fold.train_indices
        )


def test_single_recording_fails_if_30_frame_buffer_cannot_fit():
    data = _dataset([("participant_01", "session_01", 0)], samples_per_label=45)

    with pytest.raises(ValueError, match="30-frame buffer"):
        make_evaluation_splits(data)
