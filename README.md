# Face Emotion Detector

Real-time facial emotion detection using your webcam. Detects your expression and displays a matching meme overlay (currently using hamster reaction images from TikTok) side by side with the camera feed.

## Display

The window is split into two square panels:
- **Left** — live camera feed with face bounding box and emotion label
- **Right** — meme overlay matching the detected emotion (white background when no face is detected)

## Requirements

- Python 3.8+
- Webcam

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install opencv-python numpy scikit-learn
```

> **Note:** TensorFlow is not used. This project uses a scikit-learn SVM classifier, which works on Python 3.13+ and installs in seconds.

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

## License

MIT © Maria
