import numpy as np
import pytest

from training.trainer import augment_normalized_features, train_model


def test_training_augmentation_preserves_origin_and_does_not_flip_vertical_orientation():
    batch = np.zeros((4, 63), dtype=np.float32)
    points = batch.reshape((4, 21, 3))
    points[:, 1:, 1] = 0.8

    augmented = augment_normalized_features(batch, np.random.default_rng(42))
    augmented_points = augmented.reshape((4, 21, 3))

    assert augmented.shape == batch.shape
    assert np.isfinite(augmented).all()
    np.testing.assert_array_equal(augmented_points[:, 0, :], np.zeros((4, 3)))
    assert np.all(augmented_points[:, 1:, 1] > 0.6)


def test_training_fails_without_real_csv_unless_smoke_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="Collect real data"):
        train_model(data_dir=tmp_path, save_dir=tmp_path / "out")

    assert not (tmp_path / "out").exists()
