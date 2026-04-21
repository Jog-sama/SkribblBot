"""
build_features.py – Load raw Quick Draw .npy files, split into train/test,
and extract HOG features for the classical ML pipeline.

Saved artefacts (under data/processed/):
    X_train_raw.npy, y_train.npy  -> pixel arrays for deep model
    X_test_raw.npy,  y_test.npy   -> pixel arrays for evaluation
    X_train_hog.npy               -> HOG feature matrix for Random Forest
    X_test_hog.npy

Usage:
    python scripts/build_features.py
"""

import sys
from pathlib import Path

import numpy as np
from skimage.feature import hog

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLASSES,
    RAW_DIR,
    PROCESSED_DIR,
    TRAIN_SAMPLES_PER_CLASS,
    TEST_SAMPLES_PER_CLASS,
    IMG_SIZE,
    HOG_ORIENTATIONS,
    HOG_PIXELS_PER_CELL,
    HOG_CELLS_PER_BLOCK,
)


def load_class_data(cls: str, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and slice pixel data for a single class.

    The Quick Draw .npy files contain rows of 784-element uint8 vectors
    (28×28 flattened, pixel values 0–255, white stroke on black background).

    Args:
        cls:    Class name.
        n_train: Number of training samples to keep.
        n_test:  Number of test samples to keep.

    Returns:
        Tuple of (train_pixels, test_pixels) each shaped (n, 784).
    """
    path = RAW_DIR / f"{cls}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/make_dataset.py first."
        )
    data = np.load(path, mmap_mode="r")  # memory mapped for large files

    # Shuffle deterministically so splits are reproducible
    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(data))[: n_train + n_test]
    data = data[indices]

    return data[:n_train], data[n_train : n_train + n_test]


def extract_hog_features(pixel_matrix: np.ndarray) -> np.ndarray:
    """Compute HOG descriptors for a batch of flat pixel vectors.

    Args:
        pixel_matrix: Array of shape (N, 784), dtype uint8.

    Returns:
        Feature matrix of shape (N, D) where D is the HOG descriptor length.
    """
    features = []
    for row in pixel_matrix:
        img = row.reshape(IMG_SIZE, IMG_SIZE)
        desc = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=False,
            channel_axis=None,
        )
        features.append(desc)
    return np.array(features, dtype=np.float32)


def build_splits() -> None:
    """Assemble train/test raw arrays and labels from all classes."""
    train_raws, test_raws = [], []
    train_labels, test_labels = [], []

    print("Loading raw data …")
    for label_idx, cls in enumerate(CLASSES):
        print(f"  {cls} ({label_idx + 1}/{len(CLASSES)})")
        tr, te = load_class_data(cls, TRAIN_SAMPLES_PER_CLASS, TEST_SAMPLES_PER_CLASS)
        train_raws.append(tr)
        test_raws.append(te)
        train_labels.append(np.full(len(tr), label_idx, dtype=np.int64))
        test_labels.append(np.full(len(te), label_idx, dtype=np.int64))

    X_train_raw = np.concatenate(train_raws)
    X_test_raw = np.concatenate(test_raws)
    y_train = np.concatenate(train_labels)
    y_test = np.concatenate(test_labels)

    # Shuffle training set
    rng = np.random.default_rng(seed=0)
    perm = rng.permutation(len(X_train_raw))
    X_train_raw = X_train_raw[perm]
    y_train = y_train[perm]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train_raw.npy", X_train_raw)
    np.save(PROCESSED_DIR / "X_test_raw.npy", X_test_raw)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"\nSaved raw splits  →  train {X_train_raw.shape}, test {X_test_raw.shape}")


def build_hog_features() -> None:
    """Extract HOG features from saved raw arrays."""
    X_train_raw = np.load(PROCESSED_DIR / "X_train_raw.npy")
    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")

    print("Extracting HOG features (train) …")
    X_train_hog = extract_hog_features(X_train_raw)
    print("Extracting HOG features (test) …")
    X_test_hog = extract_hog_features(X_test_raw)

    np.save(PROCESSED_DIR / "X_train_hog.npy", X_train_hog)
    np.save(PROCESSED_DIR / "X_test_hog.npy", X_test_hog)
    print(f"Saved HOG features  →  train {X_train_hog.shape}, test {X_test_hog.shape}")


def build_all() -> None:
    """Run the complete feature building pipeline."""
    build_splits()
    build_hog_features()
    print("\nFeature pipeline complete.")


if __name__ == "__main__":
    build_all()
