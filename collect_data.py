import os
import pandas as pd

INPUT_FILE = "data/public_landmarks.csv"
OUTPUT_FILE = "data/gestures.csv"

print("\n======================================")
print("VISIONCONTROL DATASET CONVERTER")
print("======================================")

if not os.path.exists(INPUT_FILE):
    print("\nERROR: public_landmarks.csv not found.")
    raise SystemExit

print("\nLoading public dataset...")

df = pd.read_csv(INPUT_FILE)

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("Original dataset shape:", df.shape)

LABEL_COLUMN = "gesture_label"

if LABEL_COLUMN not in df.columns:
    print("\nERROR: gesture_label column not found.")
    print(list(df.columns))
    raise SystemExit

landmark_columns = []

for i in range(21):
    x_col = f"landmark_{i}_x"
    y_col = f"landmark_{i}_y"
    z_col = f"landmark_{i}_z"

    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
        landmark_columns.extend([x_col, y_col, z_col])

if len(landmark_columns) != 63:
    print("\nERROR: All 63 landmark columns were not found.")
    print("Found:", len(landmark_columns))
    raise SystemExit

data = df[landmark_columns + [LABEL_COLUMN]].copy()

data.rename(columns={LABEL_COLUMN: "label"}, inplace=True)

data.dropna(inplace=True)

# Clean labels
data["label"] = (
    data["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert all 12 labels to professional names
label_mapping = {
    "open": "OPEN_PALM",
    "open_inverted": "OPEN_PALM_INVERTED",
    "close": "FIST",
    "close_inverted": "FIST_INVERTED",
    "point": "POINT",
    "point_inverted": "POINT_INVERTED",
    "peace": "PEACE",
    "peace_inverted": "PEACE_INVERTED",
    "thumb": "THUMBS_UP",
    "thumb_inverted": "THUMBS_UP_INVERTED",
    "rock": "ROCK",
    "rock_inverted": "ROCK_INVERTED"
}

data = data[data["label"].isin(label_mapping.keys())].copy()

data["label"] = data["label"].map(label_mapping)

# Shuffle
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

data.to_csv(OUTPUT_FILE, index=False)

print("\n======================================")
print("CONVERSION COMPLETE")
print("======================================")

print("\nFinal dataset shape:", data.shape)

print("\nFinal class distribution:")
print(data["label"].value_counts())

print("\nSaved successfully:")
print(OUTPUT_FILE)