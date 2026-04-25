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

Install dependencies:

```bash
pip install -r requirements.txt
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

## Steps to run

### 1. Collect emotion images
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

---

### 2. Resize overlays
*(skip if overlays are already correctly sized)*

Resizes all PNGs in `overlays/` to 128×128:

```bash
python resize_overlays.py
```

---

### 3. Train the model

Reads the dataset and saves a trained SVM to `emotion_model.pkl`:

```bash
python train_emotion_model.py
```

---

### 4. Run live detection

```bash
python detect_emotion_live.py
```

Press `q` to quit.

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
├── collect_emotions.py      # webcam dataset collector
├── train_emotion_model.py   # SVM training script
├── detect_emotion_live.py   # real-time detection
├── resize_overlays.py       # overlay pre-processing utility
├── overlays/                # emotion overlay images (PNG)
├── dataset/                 # collected face images (gitignored)
├── tests/                   # pytest test suite
└── .github/workflows/ci.yml # GitHub Actions CI
```

---

## License

MIT © Maria
