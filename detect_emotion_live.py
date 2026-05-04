import cv2
import numpy as np
import pickle
import time
import os
from typing import Optional

# ── config ────────────────────────────────────────────────────────────────────
OVERLAY_DIR  = "overlays"
OVERLAY_SIZE = (128, 128)
IMG_SIZE     = (48, 48)   # used only in haar mode
DISPLAY_SIZE = 480        # each panel is DISPLAY_SIZE x DISPLAY_SIZE
TASK_PATH    = "face_landmarker.task"
TASK_URL     = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


# ── helpers ───────────────────────────────────────────────────────────────────
def load_model(model_path: str) -> tuple:
    print(f"🔍 Loading model from {model_path} ...")
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    clf      = data["model"]
    le       = data["label_encoder"]
    emotions = list(le.classes_)
    print(f"✅ Model loaded! Emotions: {emotions}")
    return clf, le, emotions


def load_overlays(emotions: list[str]) -> dict[str, np.ndarray]:
    overlays: dict[str, np.ndarray] = {}
    for emotion in emotions:
        path = os.path.join(OVERLAY_DIR, f"{emotion}.png")
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, OVERLAY_SIZE)
            overlays[emotion] = img
            print(f"🖼️  Loaded overlay for '{emotion}'")
        else:
            print(f"⚠️  No overlay found for '{emotion}' (skipping)")
    return overlays


def open_camera(index: int = 0) -> Optional[cv2.VideoCapture]:
    print("🎥 Attempting to open camera...")
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW]:
        cap = cv2.VideoCapture(index, backend)
        time.sleep(1)
        if cap.isOpened():
            print("✅ Webcam opened successfully!\n")
            return cap
        cap.release()
    print("❌ ERROR: Could not access webcam.")
    return None


def overlay_image_alpha(
    img: np.ndarray,
    overlay: np.ndarray,
    pos: tuple[int, int],
    alpha_mask: np.ndarray,
) -> None:
    x, y = pos
    h, w = overlay.shape[0], overlay.shape[1]
    if x >= img.shape[1] or y >= img.shape[0]:
        return
    h = min(h, img.shape[0] - y)
    w = min(w, img.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    overlay_roi = img[y:y+h, x:x+w]
    alpha = alpha_mask[:h, :w, None]
    img[y:y+h, x:x+w] = (
        alpha * overlay[:h, :w, :3] + (1 - alpha) * overlay_roi
    ).astype(np.uint8)


def predict_emotion(
    gray_face: np.ndarray,
    clf,
    emotions: list[str],
) -> tuple[str, float]:
    """Haar mode: raw pixel features."""
    face  = cv2.resize(gray_face, IMG_SIZE).flatten() / 255.0
    probs = clf.predict_proba([face])[0]
    idx   = np.argmax(probs)
    return emotions[idx], float(probs[idx])


def predict_emotion_landmarks(
    landmarks_flat: np.ndarray,
    clf,
    emotions: list[str],
) -> tuple[str, float]:
    """MediaPipe mode: normalised landmark features."""
    probs = clf.predict_proba([landmarks_flat])[0]
    idx   = np.argmax(probs)
    return emotions[idx], float(probs[idx])


def landmarks_to_array(face_landmarks) -> np.ndarray:
    """Convert a FaceLandmarkerResult face_landmarks[0] to a flat (478*3,) array."""
    return np.array([[p.x, p.y, p.z] for p in face_landmarks], dtype=np.float32).flatten()


def face_bbox_from_landmarks(
    face_landmarks, frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Derive a bounding box (x, y, w, h) from a list of NormalizedLandmarks."""
    xs = [p.x * frame_w for p in face_landmarks]
    ys = [p.y * frame_h for p in face_landmarks]
    x0, y0 = int(min(xs)), int(min(ys))
    x1, y1 = int(max(xs)), int(max(ys))
    return x0, y0, x1 - x0, y1 - y0


def crop_to_square(img: np.ndarray) -> np.ndarray:
    """Centre-crop an image to 1:1 aspect ratio."""
    h, w  = img.shape[:2]
    size  = min(h, w)
    y0    = (h - size) // 2
    x0    = (w - size) // 2
    return img[y0:y0+size, x0:x0+size]


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import urllib.request

    parser = argparse.ArgumentParser(description="Real-time facial emotion detector")
    parser.add_argument(
        "--detector",
        choices=["haar", "mediapipe"],
        default="haar",
        help="Face detector backend (default: haar)",
    )
    args          = parser.parse_args()
    use_mediapipe = args.detector == "mediapipe"

    if use_mediapipe:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError:
            print("❌ mediapipe is not installed. Run: pip install mediapipe")
            exit(1)

        if not os.path.exists(TASK_PATH):
            print("📥 Downloading face_landmarker.task ...")
            urllib.request.urlretrieve(TASK_URL, TASK_PATH)
            print("✅ Downloaded face_landmarker.task")

        _options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=TASK_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(_options)
        print("✅ MediaPipe FaceLandmarker ready")
    else:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    model_path       = "emotion_model_mediapipe.pkl" if use_mediapipe else "emotion_model.pkl"
    clf, le, EMOTIONS = load_model(model_path)
    overlays          = load_overlays(EMOTIONS)

    cap = open_camera()
    if cap is None:
        exit()

    blank_panel = np.full((DISPLAY_SIZE, DISPLAY_SIZE, 3), 255, dtype=np.uint8)
    mode_label  = "MediaPipe" if use_mediapipe else "Haar"
    frame_ts_ms = 0   # monotonic timestamp for VIDEO mode
    print(f"🤖 Real-time Emotion Detection started [{mode_label}] (Press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Lost camera feed! Retrying...")
            cap.release()
            cap = open_camera()
            if cap is None:
                print("❌ Could not reconnect. Exiting...")
                break
            continue

        frame           = cv2.flip(frame, 1)
        current_overlay = blank_panel

        if use_mediapipe:
            h_frame, w_frame = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_ts_ms += 33   # ~30 fps
            result   = landmarker.detect_for_video(mp_image, frame_ts_ms)

            if result.face_landmarks:
                lm_list  = result.face_landmarks[0]
                landmarks = landmarks_to_array(lm_list)
                emotion, confidence = predict_emotion_landmarks(landmarks, clf, EMOTIONS)

                x, y, w, h = face_bbox_from_landmarks(lm_list, w_frame, h_frame)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if emotion in overlays:
                    current_overlay = cv2.resize(
                        overlays[emotion][:, :, :3], (DISPLAY_SIZE, DISPLAY_SIZE)
                    )
        else:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_gray           = gray[y:y+h, x:x+w]
                emotion, confidence = predict_emotion(face_gray, clf, EMOTIONS)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                if emotion in overlays:
                    current_overlay = cv2.resize(
                        overlays[emotion][:, :, :3], (DISPLAY_SIZE, DISPLAY_SIZE)
                    )

        cam_panel = cv2.resize(crop_to_square(frame), (DISPLAY_SIZE, DISPLAY_SIZE))
        cv2.imshow("Emotion Detector", np.hstack([cam_panel, current_overlay]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    if use_mediapipe:
        landmarker.close()
