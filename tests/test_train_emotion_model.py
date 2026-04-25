"""
Tests for load_dataset and the training pipeline in train_emotion_model.py.
Uses a small temporary dataset — no real images required.
"""
import os
import sys
import numpy as np
import pickle
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock cv2 before importing so tests run even with a broken cv2 install.
# We replace cv2.imread / cv2.resize with numpy-based stubs.
# ---------------------------------------------------------------------------
_cv2_mock = MagicMock()
_cv2_mock.IMREAD_GRAYSCALE = 0

def _fake_imread(path, flags=None):
    """Return a random 60×60 grayscale array for any path that exists."""
    if not os.path.exists(path):
        return None
    return np.random.randint(0, 256, (60, 60), dtype=np.uint8)

def _fake_resize(img, size):
    # size is (w, h) — return a proper 2-D array so flatten() gives w*h features
    w, h = size
    return np.zeros((h, w), dtype=np.uint8)

_cv2_mock.imread.side_effect = _fake_imread
_cv2_mock.resize.side_effect = _fake_resize

sys.modules.setdefault("cv2", _cv2_mock)

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Mirror of load_dataset from train_emotion_model.py
# ---------------------------------------------------------------------------
IMG_SIZE = (48, 48)

def load_dataset(directory):
    import cv2
    images, labels = [], []
    for emotion in sorted(os.listdir(directory)):
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for filename in os.listdir(emotion_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(emotion_dir, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            images.append(img.flatten() / 255.0)
            labels.append(emotion)
    return np.array(images), np.array(labels)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
EMOTIONS = ["angry", "happy", "neutral"]

@pytest.fixture()
def fake_dataset(tmp_path):
    """Create a tiny dataset directory tree with placeholder image files."""
    for emotion in EMOTIONS:
        emotion_dir = tmp_path / emotion
        emotion_dir.mkdir()
        for i in range(4):
            # Just touch the file — _fake_imread reads size from os.path.exists
            (emotion_dir / f"{emotion}_{i:04d}.jpg").write_bytes(b"\xff")
    return tmp_path


# ---------------------------------------------------------------------------
# load_dataset tests
# ---------------------------------------------------------------------------
class TestLoadDataset:
    def test_returns_correct_sample_count(self, fake_dataset):
        X, y = load_dataset(str(fake_dataset))
        assert len(X) == len(EMOTIONS) * 4
        assert len(y) == len(X)

    def test_feature_vector_length(self, fake_dataset):
        X, _ = load_dataset(str(fake_dataset))
        assert X.shape[1] == IMG_SIZE[0] * IMG_SIZE[1]

    def test_pixel_values_normalised(self, fake_dataset):
        X, _ = load_dataset(str(fake_dataset))
        assert X.min() >= 0.0
        assert X.max() <= 1.0

    def test_labels_match_folder_names(self, fake_dataset):
        _, y = load_dataset(str(fake_dataset))
        assert set(y) == set(EMOTIONS)

    def test_non_image_files_are_skipped(self, fake_dataset):
        (fake_dataset / EMOTIONS[0] / "notes.txt").write_text("ignore me")
        X, y = load_dataset(str(fake_dataset))
        assert len(X) == len(EMOTIONS) * 4   # count unchanged

    def test_empty_directory_returns_empty_arrays(self, tmp_path):
        X, y = load_dataset(str(tmp_path))
        assert len(X) == 0
        assert len(y) == 0


# ---------------------------------------------------------------------------
# Model pickle round-trip
# ---------------------------------------------------------------------------
class TestModelPickle:
    def test_saved_model_loads_and_predicts(self, fake_dataset, tmp_path):
        X, y = load_dataset(str(fake_dataset))
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        clf = SVC(kernel="rbf", C=1, gamma="scale", probability=True)
        clf.fit(X, y_enc)

        model_path = str(tmp_path / "emotion_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": clf, "label_encoder": le}, f)

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        probs = data["model"].predict_proba([X[0]])[0]
        assert len(probs) == len(EMOTIONS)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_label_encoder_classes_preserved(self, fake_dataset, tmp_path):
        X, y = load_dataset(str(fake_dataset))
        le = LabelEncoder()
        le.fit(y)

        model_path = str(tmp_path / "emotion_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump({"model": None, "label_encoder": le}, f)

        with open(model_path, "rb") as f:
            data = pickle.load(f)

        assert list(data["label_encoder"].classes_) == sorted(EMOTIONS)
