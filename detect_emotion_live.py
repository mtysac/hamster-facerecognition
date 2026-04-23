import cv2
import numpy as np
import pickle
import time
import os

# config
MODEL_PATH   = "emotion_model.pkl"
OVERLAY_DIR  = "overlays"
OVERLAY_SIZE = (128, 128)
IMG_SIZE     = (48, 48)

# load model
print(f"🔍 Loading model from {MODEL_PATH} ...")
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
clf = data["model"]
le  = data["label_encoder"]
EMOTIONS = list(le.classes_)
print(f"✅ Model loaded! Emotions: {EMOTIONS}")

# load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# load overlays
overlays = {}
for emotion in EMOTIONS:
    path = os.path.join(OVERLAY_DIR, f"{emotion}.png")
    if os.path.exists(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, OVERLAY_SIZE)
        overlays[emotion] = img
        print(f"🖼️  Loaded overlay for '{emotion}'")
    else:
        print(f"⚠️  No overlay found for '{emotion}' (skipping)")

# camera handler
def open_camera(index=0):
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

def overlay_image_alpha(img, overlay, pos, alpha_mask):
    x, y = pos
    h, w = overlay.shape[0], overlay.shape[1]
    # clamp to frame bounds
    if x >= img.shape[1] or y >= img.shape[0]:
        return
    h = min(h, img.shape[0] - y)
    w = min(w, img.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    overlay_roi = img[y:y+h, x:x+w]
    alpha = alpha_mask[:h, :w, None]
    img[y:y+h, x:x+w] = (alpha * overlay[:h, :w, :3] + (1 - alpha) * overlay_roi).astype(np.uint8)

def predict_emotion(gray_face):
    face = cv2.resize(gray_face, IMG_SIZE).flatten() / 255.0
    probs = clf.predict_proba([face])[0]
    idx = np.argmax(probs)
    return EMOTIONS[idx], probs[idx]

DISPLAY_SIZE = 480   # each panel is DISPLAY_SIZE x DISPLAY_SIZE

def crop_to_square(img):
    """Centre-crop an image to 1:1 aspect ratio."""
    h, w = img.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0+size, x0:x0+size]

cap = open_camera()
if cap is None:
    exit()

# blank right panel shown when no face / no overlay is detected
blank_panel = np.full((DISPLAY_SIZE, DISPLAY_SIZE, 3), 255, dtype=np.uint8)

print("🤖 Real-time Emotion Detection started! (Press 'q' to quit)")
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

    # flip horizontally so it behaves like a mirror
    frame = cv2.flip(frame, 1)

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_overlay = blank_panel  # default right panel

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if emotion in overlays:
            overlay_img = overlays[emotion]
            # build a square BGR panel from the overlay
            panel = cv2.resize(overlay_img[:, :, :3], (DISPLAY_SIZE, DISPLAY_SIZE))
            current_overlay = panel

    # ── build side-by-side display ────────────────────────────────────────────
    cam_square   = crop_to_square(frame)
    cam_panel    = cv2.resize(cam_square, (DISPLAY_SIZE, DISPLAY_SIZE))
    right_panel  = current_overlay

    combined = np.hstack([cam_panel, right_panel])
    cv2.imshow("Emotion Detector", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
