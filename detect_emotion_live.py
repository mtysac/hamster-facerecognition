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

cap = open_camera()
if cap is None:
    exit()

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

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_gray)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if emotion in overlays:
            overlay_img = overlays[emotion]
            if overlay_img.shape[2] == 4:
                alpha = overlay_img[:, :, 3] / 255.0
            else:
                alpha = np.ones(overlay_img.shape[:2], dtype=float)

            oy = max(y - OVERLAY_SIZE[1] - 10, 0)
            ox = min(x + w + 10, frame.shape[1] - OVERLAY_SIZE[0])
            overlay_image_alpha(frame, overlay_img, (ox, oy), alpha)

    cv2.imshow("Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
