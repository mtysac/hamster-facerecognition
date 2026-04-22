import os
import cv2
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"
IMG_SIZE  = (48, 48)
MODEL_PATH = "emotion_model.pkl"

def load_dataset(directory):
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

print("📂 Loading training data...")
X_train, y_train = load_dataset(TRAIN_DIR)
print(f"   {len(X_train)} training samples")

print("📂 Loading test data...")
X_test, y_test = load_dataset(TEST_DIR)
print(f"   {len(X_test)} test samples")

# encode labels
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

# save model + label encoder together
with open(MODEL_PATH, "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)
print(f"💾 Model saved to {MODEL_PATH}")
