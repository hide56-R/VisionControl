# VisionControl — AI Hand Gesture Recognition

VisionControl is a computer-vision-based application that recognizes hand gestures in real time using a webcam and performs predefined computer-control actions.

## Features

* Real-time hand gesture recognition
* Webcam-based computer vision
* Machine learning classification
* Gesture smoothing for stable predictions
* Confidence thresholding
* Modular action/controller architecture
* Keyboard and system-control interactions
* Clean project structure

## Supported Gestures

* FIST
* OPEN_PALM
* PEACE
* POINT
* THUMBS_UP

## Technologies

* Python
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Scikit-learn
* Random Forest
* Joblib

## Project Structure

```text
VisionControl/
│
├── recognize.py
├── actions.py
├── collect_data.py
├── train_model.py
├── requirements.txt
│
├── app/
│   └── controller.py
│
├── data/
│   └── gestures.csv
│
├── models/
│   └── gesture_model.pkl
│
└── README.md
```

## How It Works

1. Webcam captures the user's hand.
2. Hand landmarks are extracted.
3. Landmark data is processed into model features.
4. The trained machine-learning model predicts the gesture.
5. A confidence threshold and smoothing system stabilize the prediction.
6. The corresponding action is triggered.

## Running the Project

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run VisionControl:

```bash
python recognize.py
```

## Controls

* `Q` — Quit
* `R` — Reset gesture history
* `H` — Show help

## Machine Learning

The project uses a Random Forest classifier trained on hand-landmark features extracted from gesture samples.

## Purpose

VisionControl was developed as a portfolio project to demonstrate practical skills in:

* Computer Vision
* Machine Learning
* Python
* Real-time AI applications
* Software architecture
* Human-computer interaction

## Author

**Rehan Khan Afridi**

GitHub: **hide56-R** 
