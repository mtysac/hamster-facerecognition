# Face recognition with OpenCV

The images in this references memes! (right now the program is using the hamster reactions from tiktok)

## Instructions

### Note on TensorFlow
This project originally used TensorFlow for the emotion classification model. TensorFlow does not support Python 3.13 or newer — if you try to install it you'll get:

```
ERROR: Could not find a version that satisfies the requirement tensorflow
ERROR: No matching distribution found for tensorflow
```

To fix this, the project has been rewritten to use **scikit-learn** (SVM classifier) instead. It works on any Python version, installs in seconds, and produces a `.pkl` model file instead of `.h5`.

---

### Setup

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

Install dependencies:

```bash
pip install opencv-python numpy scikit-learn
```

---

### Steps to run

**1. Collect emotion images** (skip if dataset already exists)

Opens your webcam:

```bash
python collect_emotions.py
```

Press a key to capture a 5-second burst for that emotion:
- `n` → neutral
- `h` → happy
- `s` → sad
- `a` → angry
- `u` → surprise
- `q` → quit



**2. Resize overlays** (skip if overlays are already sized correctly)

Resizes all PNGs in the `overlays/` folder to 128×128:

```bash
python resize_overlays.py
```

**3. Train the model**

Reads the dataset and saves a trained SVM model to `emotion_model.pkl`:

```bash
python train_emotion_model.py
```

**4. Run live detection**

Opens your webcam and detects emotions in real time with overlay images. Press `q` to quit:

```bash
python detect_emotion_live.py
```