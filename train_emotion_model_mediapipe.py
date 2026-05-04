"""
Train an SVM emotion classifier using MediaPipe FaceLandmarker (Tasks API).
Produces emotion_model_mediapipe.pkl — used by detect_emotion_live.py --detector mediapipe

Requires:
  pip install mediapipe

Also requires the face_landmarker.task model file in the project root.
Download it with:
  python -c "import urllib.request; urllib.request.urlretrieve(
      'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
      'face_landmarker.task')"
"""
import os
import sys
import urllib.request
import cv2
import numpy as np
import pickle

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌ mediapipe is not installed. Run: pip install mediapipe")
    sys.exit(1)

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

TRAIN_DIR   = "dataset/train"
TEST_DIR    = "dataset/test"
MODEL_PATH  = "emotion_model_mediapipe.pkl"
TASK_PATH   = "face_landmarker.task"
TASK_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ── download model if missing ─────────────────────────────────────────────────
if not os.path.exists(TASK_PATH):
    print(f"📥 Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    print("✅ Downloaded face_landmarker.task")

# ── set up landmarker (IMAGE mode for static files) ───────────────────────────
BaseOptions          = mp_python.BaseOptions
FaceLandmarker       = mp_vision.FaceLandmarker
FaceLandmarkerOptions = mp_vision.FaceLandmarkerOptions
VisionRunningMode    = mp_vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=TASK_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.3,
)
landmarker = FaceLandmarker.create_from_options(options)


def extract_landmarks(image_path: str) -> np.ndarray | None:
    """Return flat (478*3,) normalised landmark array, or None if no face found."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32).flatten()


def load_dataset(directory: str) -> tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    skipped = 0
    for emotion in sorted(os.listdir(directory)):
        emotion_dir = os.path.join(directory, emotion)
        if not os.path.isdir(emotion_dir):
            continue
        for filename in os.listdir(emotion_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            landmarks = extract_landmarks(os.path.join(emotion_dir, filename))
            if landmarks is None:
                skipped += 1
                continue
            images.append(landmarks)
            labels.append(emotion)
    if skipped:
        print(f"   ⚠️  Skipped {skipped} images (no face detected)")
    return np.array(images), np.array(labels)


print("📂 Loading training data (extracting landmarks)...")
X_train, y_train = load_dataset(TRAIN_DIR)
print(f"   {len(X_train)} training samples, {X_train.shape[1] if len(X_train) else 0} features each")

print("📂 Loading test data (extracting landmarks)...")
X_test, y_test = load_dataset(TEST_DIR)
print(f"   {len(X_test)} test samples")

if len(X_train) == 0:
    print("❌ No training samples found. Check your dataset directory.")
    landmarker.close()
    sys.exit(1)

le = LabelEncoder()
le.fit(np.concatenate([y_train, y_test]))
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

print("\n🧠 Training SVM classifier...")
clf = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
clf.fit(X_train, y_train_enc)
print("✅ Training complete!")

print("\n📊 Evaluation on test set:")
y_pred = clf.predict(X_test)
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)
print(f"💾 Model saved to {MODEL_PATH}")

landmarker.close()
