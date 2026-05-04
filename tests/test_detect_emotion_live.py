"""
Tests for pure utility functions in detect_emotion_live.py.
No webcam, no model file, no real images required.
"""
import numpy as np
import pytest
import pickle
import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Build a real SVC trained on 2304-feature vectors (48*48) so predict_emotion
# works correctly without a saved model file.
# ---------------------------------------------------------------------------
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

EMOTIONS = ["angry", "happy", "neutral", "sad", "surprise"]

def _make_fake_clf():
    le = LabelEncoder()
    le.fit(EMOTIONS)
    clf = SVC(probability=True)
    n_features = 48 * 48
    X = np.eye(len(EMOTIONS), n_features)
    clf.fit(X, le.transform(le.classes_))
    return clf

FAKE_CLF = _make_fake_clf()

# ---------------------------------------------------------------------------
# Mock cv2 before importing so the module loads without a working cv2 install.
# resize must return a proper 2-D array so .flatten() gives 48*48 features.
# ---------------------------------------------------------------------------
_cv2_mock = MagicMock()
_cv2_mock.data.haarcascades = ""

def _fake_cv2_resize(img, size, **kwargs):
    # size is (w, h) — return shape (h, w)
    return np.zeros(size[::-1], dtype=np.uint8)

_cv2_mock.resize.side_effect = _fake_cv2_resize
sys.modules.setdefault("cv2", _cv2_mock)

import detect_emotion_live as del_module


# ---------------------------------------------------------------------------
# crop_to_square
# ---------------------------------------------------------------------------
class TestCropToSquare:
    def test_landscape_becomes_square(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = del_module.crop_to_square(img)
        h, w = result.shape[:2]
        assert h == w == 480

    def test_portrait_becomes_square(self):
        img = np.zeros((640, 480, 3), dtype=np.uint8)
        result = del_module.crop_to_square(img)
        h, w = result.shape[:2]
        assert h == w == 480

    def test_already_square_unchanged(self):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        result = del_module.crop_to_square(img)
        assert result.shape == (300, 300, 3)

    def test_centre_crop_preserves_content(self):
        """The centre 100×100 region of a 100×200 image should survive."""
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        img[:, 50:150] = 255
        result = del_module.crop_to_square(img)
        assert result.shape == (100, 100, 3)
        assert np.all(result == 255)

    def test_output_is_3d(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = del_module.crop_to_square(img)
        assert result.ndim == 3


# ---------------------------------------------------------------------------
# predict_emotion  (haar mode — raw pixel features)
# ---------------------------------------------------------------------------
class TestPredictEmotion:
    def test_returns_known_emotion(self):
        face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        emotion, _ = del_module.predict_emotion(face, FAKE_CLF, EMOTIONS)
        assert emotion in EMOTIONS

    def test_confidence_in_range(self):
        face = np.random.randint(0, 256, (48, 48), dtype=np.uint8)
        _, confidence = del_module.predict_emotion(face, FAKE_CLF, EMOTIONS)
        assert 0.0 <= confidence <= 1.0

    def test_accepts_various_input_sizes(self):
        """predict_emotion resizes internally — any 2-D array should work."""
        for size in [(24, 24), (48, 48), (96, 96)]:
            face = np.zeros(size, dtype=np.uint8)
            emotion, _ = del_module.predict_emotion(face, FAKE_CLF, EMOTIONS)
            assert emotion in EMOTIONS

    def test_returns_two_element_tuple(self):
        face = np.zeros((48, 48), dtype=np.uint8)
        result = del_module.predict_emotion(face, FAKE_CLF, EMOTIONS)
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# predict_emotion_landmarks  (mediapipe mode — landmark features)
# ---------------------------------------------------------------------------
class TestPredictEmotionLandmarks:
    def _make_landmark_clf(self):
        """SVC trained on 478*3=1434 landmark features."""
        le = LabelEncoder()
        le.fit(EMOTIONS)
        clf = SVC(probability=True)
        n_features = 478 * 3
        X = np.eye(len(EMOTIONS), n_features)
        clf.fit(X, le.transform(le.classes_))
        return clf

    def test_returns_known_emotion(self):
        clf = self._make_landmark_clf()
        landmarks = np.zeros(478 * 3, dtype=np.float32)
        emotion, _ = del_module.predict_emotion_landmarks(landmarks, clf, EMOTIONS)
        assert emotion in EMOTIONS

    def test_confidence_in_range(self):
        clf = self._make_landmark_clf()
        landmarks = np.random.rand(478 * 3).astype(np.float32)
        _, confidence = del_module.predict_emotion_landmarks(landmarks, clf, EMOTIONS)
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# overlay_image_alpha
# ---------------------------------------------------------------------------
class TestOverlayImageAlpha:
    def _make_overlay(self, h=64, w=64, alpha=255):
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[:, :, :3] = 200
        overlay[:, :, 3] = alpha
        return overlay

    def test_opaque_overlay_replaces_background(self):
        bg = np.zeros((128, 128, 3), dtype=np.uint8)
        overlay = self._make_overlay(64, 64, alpha=255)
        alpha_mask = overlay[:, :, 3] / 255.0
        del_module.overlay_image_alpha(bg, overlay, (0, 0), alpha_mask)
        assert np.all(bg[0:64, 0:64] == 200)

    def test_transparent_overlay_leaves_background(self):
        bg = np.full((128, 128, 3), 100, dtype=np.uint8)
        overlay = self._make_overlay(64, 64, alpha=0)
        alpha_mask = overlay[:, :, 3] / 255.0
        del_module.overlay_image_alpha(bg, overlay, (0, 0), alpha_mask)
        assert np.all(bg[0:64, 0:64] == 100)

    def test_out_of_bounds_position_does_not_crash(self):
        bg = np.zeros((128, 128, 3), dtype=np.uint8)
        overlay = self._make_overlay(64, 64)
        alpha_mask = overlay[:, :, 3] / 255.0
        del_module.overlay_image_alpha(bg, overlay, (200, 200), alpha_mask)

    def test_partial_overlap_clamps_correctly(self):
        bg = np.zeros((128, 128, 3), dtype=np.uint8)
        overlay = self._make_overlay(64, 64, alpha=255)
        alpha_mask = overlay[:, :, 3] / 255.0
        del_module.overlay_image_alpha(bg, overlay, (100, 100), alpha_mask)
        assert np.all(bg[100:128, 100:128] == 200)
        assert np.all(bg[0:100, 0:100] == 0)
