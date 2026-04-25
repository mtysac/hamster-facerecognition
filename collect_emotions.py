import cv2
import os
import time
import random

# base directory
BASE_DIR = 'dataset'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')

# define emotions + key binding
EMOTIONS = {
    'n' : 'neutral',
    'h' : 'happy',
    's' : 'sad',
    'a' : 'angry',
    'u' : 'surprise' # u -> surprise cause s -> sad
}

MAX_IMAGES_PER_EMOTION = 100
CAPTURE_TIME = 5
FPS_TARGET = 4

# create directories
for subset in [TRAIN_DIR, TEST_DIR]:
    for emotion in EMOTIONS.values():
        os.makedirs(os.path.join(subset, emotion), exist_ok = True)

# webcam setup
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print('✅ Webcam ready! Press:')
for k, v in EMOTIONS.items():
    print(f" '{k}' -> {v}")
print('Press q to quit.\n')

counts = {e: len(os.listdir(os.path.join(TRAIN_DIR, e))) + len(os.listdir(os.path.join(TEST_DIR, e)))
          for e in EMOTIONS.values()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade. detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.putText(frame, 'Press (n/h/s/a/u) or "q" to quit', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    y_offset = 60
    for emotion, count in counts.items():
        cv2.putText(frame, f'{emotion}: {count}/{MAX_IMAGES_PER_EMOTION}', (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        y_offset += 25

    cv2.imshow('Emotion Dataset Collector', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("👋 Exiting...")
        break
    for k, emotion  in EMOTIONS.items():
        if key == ord(k):
            if counts[emotion] >= MAX_IMAGES_PER_EMOTION:
                print(f'"{emotion}" already has 100 images')
                continue

            print(f'\nCapturing {emotion} images for {CAPTURE_TIME} seconds...')
            start_time = time.time()

            while time.time() - start_time < CAPTURE_TIME and counts[emotion] < MAX_IMAGES_PER_EMOTION:
                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (48, 48))

                    # 80% train 20% test
                    subset = TRAIN_DIR if random.random() > 0.2 else TEST_DIR

                    counts[emotion] += 1
                    filename = f'{emotion}_{counts[emotion]:04d}.jpg'
                    save_path = os.path.join(subset, emotion, filename)
                    cv2.imwrite(save_path, face_img)

                cv2.imshow('Emotion Dataset Collector', frame)
                if cv2.waitKey(int(1000 / FPS_TARGET)) & 0xFF == ord('q'):
                    break

            print(f'✅ Done capturing for {emotion} ({counts[emotion]}/100\n)')
cap.release()
cv2.destroyAllWindows()