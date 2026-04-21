"""
Central configuration for ScribblBot.
All hyperparameters, paths, and constants live here.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in [RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Classes
# 15 visually distinct Quick Draw categories
CLASSES = [
    "cat", "dog", "pizza", "bicycle", "house",
    "sun", "tree", "car", "fish", "butterfly",
    "guitar", "hamburger", "airplane", "banana", "star",
]
NUM_CLASSES = len(CLASSES)

CLASS_EMOJIS = {
    "cat": "🐱", "dog": "🐶", "pizza": "🍕", "bicycle": "🚲",
    "house": "🏠", "sun": "☀️", "tree": "🌳", "car": "🚗",
    "fish": "🐟", "butterfly": "🦋", "guitar": "🎸", "hamburger": "🍔",
    "airplane": "✈️", "banana": "🍌", "star": "⭐",
}

# Dataset
TRAIN_SAMPLES_PER_CLASS = 2000   # keeps training fast (~30k total)
TEST_SAMPLES_PER_CLASS = 400     # solid eval set (~6k total)
IMG_SIZE = 28                    # Quick Draw native resolution

# Quick Draw public GCS bucket
QUICKDRAW_URL = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{cls}.npy"
)

# Deep Model
DEEP_BATCH_SIZE = 128
DEEP_EPOCHS = 15
DEEP_LR = 1e-3
DEEP_WEIGHT_DECAY = 1e-4

# Classical Model
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (4, 4)
HOG_CELLS_PER_BLOCK = (2, 2)

# Experiment: training set size sensitivity
EXPERIMENT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]
EXPERIMENT_EPOCHS = 10     # shorter runs for the sweep
