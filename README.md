# VisionControl

VisionControl is a Python computer-vision desktop-control application. It uses a
webcam, MediaPipe hand landmarks, and a scikit-learn Random Forest model to turn
recognised hand gestures into media and mouse commands.

## Features

- Real-time hand tracking and gesture recognition.
- Shared wrist-relative, scale-normalised 63-feature pipeline for training and inference.
- Configurable confidence filtering; low-confidence gestures do not execute actions.
- Media mode, mouse mode, mouse safety toggle, action history, FPS, and dashboard.
- A separate Training Studio for building a personal gesture dataset safely.

## Gesture controls

| Gesture | Media mode | Mouse mode |
| --- | --- | --- |
| THUMBS_UP | Volume up | - |
| THUMBS_UP_INVERTED | Volume down | - |
| FIST | Play / pause | Left click |
| PEACE | Screenshot | Double click |
| POINT | Next track | Move cursor |
| OPEN_PALM | Stop media | - |
| ROCK | Previous track | Right click |

## Install and run

```powershell
cd "C:\Users\Rehan Afridii\VisionControl"
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\.venv\Scripts\python.exe recognize.py
```

Application controls: `M` switches Media/Mouse mode, `X` toggles mouse control,
`G` toggles recognition, `R` resets actions, `H` shows help, and `Q` quits.

## Media mode and mouse safety

Media mode is the default. It sends the mapped media action only after the model
passes the configured confidence threshold and the action cooldown allows it.
Removing the hand clears the media gesture latch, so the same gesture can be used
again after the cooldown without repeating continuously while held.

Mouse mode must be selected with `M`, and mouse control is still **OFF** until
`X` enables it. Cursor movement, clicks, and double-clicks are never sent while
mouse control is OFF, recognition is disabled, no hand is visible, or confidence
is too low. Use `X` to turn mouse control OFF before moving away from the computer.

## Collecting Your Own Dataset

The Training Studio stores your personal samples in `data/gesture_dataset.csv`.
It preserves any existing rows, checks the 63-feature schema before appending, and
does not affect the current production model.

1. Start the collector.
2. Press `1` to `7` to select a gesture.
3. Press `S` to start collection.
4. Hold that gesture from varied positions, angles, distances, and lighting.
5. Wait until the dashboard shows `200 / 200` and `TARGET REACHED`.
6. Repeat for every gesture, keeping counts balanced.
7. Run statistics.
8. Train and evaluate a candidate model.
9. Promote only when its metrics and balance are satisfactory.

```powershell
# Collect: 1-7 select, S start/pause, R reset state, H help, Q quit
.\.venv\Scripts\python.exe tools\collect_data.py

# See actual totals, minimum/maximum class size, and imbalance warnings
.\.venv\Scripts\python.exe tools\dataset_stats.py

# Train a candidate only: models/gesture_model_new.pkl
.\.venv\Scripts\python.exe train_model.py

# Evaluate the candidate against the held-out split
.\.venv\Scripts\python.exe tools\evaluate_model.py

# Promote only after review. The production model is backed up first.
.\.venv\Scripts\python.exe tools\promote_model.py
```

The collector captures one valid hand at approximately 120 ms intervals, rejects
invalid feature vectors, and skips exact duplicate vectors. `R` resets only the
current collection state; it never deletes saved data. The model is not perfect:
the statistics and training tools warn about small or imbalanced classes.

## Recognition and ML pipeline

1. OpenCV reads a mirrored webcam frame.
2. MediaPipe produces 21 ordered `(x, y, z)` hand landmarks.
3. `feature_extraction.py` subtracts the wrist and scale-normalises the 63 values.
4. Random Forest predicts a gesture and confidence.
5. Gestures below the configured confidence threshold become `UNKNOWN`.

The exact same feature-extraction module is used by live recognition, data
collection, and training.

## Known limitations

- The bundled production dataset has 364 samples and class sizes from 17 to 40.
  It is suitable for demonstrating the pipeline, but live accuracy depends on
  camera angle, lighting, hand shape, and distance.
- The reported evaluation is a split/cross-validation estimate on that dataset,
  not a guarantee of real-world accuracy.
- Some inverted model classes intentionally have no media action; they remain
  visible predictions and are never silently remapped to another gesture.

## Architecture

```text
app/                    Runtime application modules
tools/                  Dataset collection, statistics, evaluation, promotion
data/                   Landmark datasets
models/                 Production and candidate model files
feature_extraction.py   Shared ML feature pipeline
train_model.py          Candidate-model training only
recognize.py            Application entry point
actions.py              Media action mappings
mouse_control.py        Mouse primitives
```
