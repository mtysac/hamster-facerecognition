# Face Emotion Detector

Real-time facial emotion detection using your webcam. Detects your expression and displays a matching meme overlay (currently using hamster reaction images from TikTok) side by side with the camera feed.

![CI](https://github.com/MTYSAC/hamster-facerecognition/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Display

The window is split into two square panels:
- **Left** — live camera feed with face bounding box and emotion label
- **Right** — meme overlay matching the detected emotion (white background when no face is detected)

## Requirements

- Python 3.8–3.13 (opencv-python does not yet support 3.14+)
- Webcam

Install core dependencies:

```bash
pip install -r requirements.txt
```

For the MediaPipe detector (optional):

```bash
pip install mediapipe
```

> **Note:** TensorFlow is not used. This project uses a scikit-learn SVM classifier, which installs in seconds.

---

## Setup

Optionally create a virtual environment first:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
.\venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

## Two detector modes

| Mode | Flag | Face detection | Features | Model file |
|------|------|---------------|----------|------------|
| Haar (default) | *(none)* | OpenCV Haar cascade | Raw pixels (48×48) | `emotion_model.pkl` |
| MediaPipe | `--detector mediapipe` | MediaPipe FaceLandmarker (Tasks API) | 478 landmarks × 3 | `emotion_model_mediapipe.pkl` |

Both modes use the same SVM pipeline and overlay display. MediaPipe is generally more accurate and robust to lighting, but requires an extra install and a separately trained model.

---

## Steps to run

### Option A — own dataset (Haar)

#### 1. Collect emotion images
*(skip if the `dataset/` folder already exists)*

Opens your webcam. Press a key to capture a 5-second burst for that emotion:

```bash
python collect_emotions.py
```

| Key | Emotion  |
|-----|----------|
| `n` | neutral  |
| `h` | happy    |
| `s` | sad      |
| `a` | angry    |
| `u` | surprise |
| `q` | quit     |

#### 2. Resize overlays
*(skip if overlays are already correctly sized)*

```bash
python resize_overlays.py
```

#### 3. Train the model

```bash
python train_emotion_model.py
```

Saves `emotion_model.pkl`.

#### 4. Run live detection

```bash
python detect_emotion_live.py
```

---

### Option B — MediaPipe landmarks

Requires the same dataset collected in Option A (MediaPipe extracts landmarks from those images instead of raw pixels).

#### 1. Install MediaPipe

```bash
pip install mediapipe
```

#### 2. Train the MediaPipe model

```bash
python train_emotion_model_mediapipe.py
```

Saves `emotion_model_mediapipe.pkl`. The script auto-downloads `face_landmarker.task` (~30 MB) on first run.

#### 3. Run live detection

```bash
python detect_emotion_live.py --detector mediapipe
```

---

Press `q` to quit either mode.

---

## Testing

Run the test suite (no webcam or trained model required):

```bash
pip install pytest
pytest tests/ -v
```

Tests cover `crop_to_square`, `predict_emotion`, `overlay_image_alpha`, `load_dataset`, and the model pickle round-trip. CI runs automatically on every push via GitHub Actions.

---

## Project structure

```
├── collect_emotions.py               # webcam dataset collector
├── train_emotion_model.py            # SVM training (Haar / raw pixels)
├── train_emotion_model_mediapipe.py  # SVM training (MediaPipe landmarks)
├── detect_emotion_live.py            # real-time detection (--detector haar|mediapipe)
├── resize_overlays.py                # overlay pre-processing utility
├── overlays/                         # emotion overlay images (PNG)
├── dataset/                          # collected face images (gitignored)
├── tests/                            # pytest test suite
└── .github/workflows/ci.yml          # GitHub Actions CI
```

---

## License

MIT © Maria Cabrera
