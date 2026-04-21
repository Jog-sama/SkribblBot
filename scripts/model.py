"""
model.py – Define, train, and evaluate all three models:
    1. Naive baseline (majority class classifier)
    2. Classical ML  (Random Forest on HOG features)
    3. Deep learning (ScribblNet CNN)

Also runs the training size sensitivity experiment and saves results/plots.

Usage:
    python scripts/model.py
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    CLASSES,
    MODELS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    NUM_CLASSES,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    DEEP_BATCH_SIZE,
    DEEP_EPOCHS,
    DEEP_LR,
    DEEP_WEIGHT_DECAY,
    IMG_SIZE,
    EXPERIMENT_FRACTIONS,
    EXPERIMENT_EPOCHS,
)


# Utility

def get_device() -> torch.device:
    """Return the best available torch device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_processed_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load all processed arrays from disk.

    Returns:
        X_train_raw, X_test_raw, y_train, y_test, X_train_hog, X_test_hog
    """
    X_train_raw = np.load(PROCESSED_DIR / "X_train_raw.npy")
    X_test_raw = np.load(PROCESSED_DIR / "X_test_raw.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    X_train_hog = np.load(PROCESSED_DIR / "X_train_hog.npy")
    X_test_hog = np.load(PROCESSED_DIR / "X_test_hog.npy")
    return X_train_raw, X_test_raw, y_train, y_test, X_train_hog, X_test_hog


# 1. Naive Baseline

class MajorityClassifier:
    """Naive baseline: always predicts the most frequent class in training."""

    def __init__(self) -> None:
        self.majority_class: int = 0

    def fit(self, y: np.ndarray) -> "MajorityClassifier":
        """Fit by finding the majority class label.

        Args:
            y: 1-D array of integer class labels.

        Returns:
            self
        """
        counts = np.bincount(y)
        self.majority_class = int(np.argmax(counts))
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        """Return the majority class repeated n_samples times.

        Args:
            n_samples: Number of predictions to generate.

        Returns:
            Array of length n_samples, all equal to majority_class.
        """
        return np.full(n_samples, self.majority_class, dtype=np.int64)


def train_naive(y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Train and evaluate the majority class baseline.

    Args:
        y_train: Training labels.
        y_test:  Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"\nNaive Baseline")
    clf = MajorityClassifier().fit(y_train)
    preds = clf.predict(len(y_test))
    acc = accuracy_score(y_test, preds)
    print(f"  Majority class: {CLASSES[clf.majority_class]}")
    print(f"  Test accuracy:  {acc:.4f}")

    model_data = {"majority_class": clf.majority_class, "accuracy": acc}
    joblib.dump(model_data, MODELS_DIR / "naive_model.pkl")

    return {"model": "naive", "accuracy": acc}


# 2. Classical ML

def train_classical(
    X_train_hog: np.ndarray,
    X_test_hog: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Train Random Forest on HOG features and evaluate.

    Args:
        X_train_hog: Training HOG feature matrix.
        X_test_hog:  Test HOG feature matrix.
        y_train:     Training labels.
        y_test:      Test labels.

    Returns:
        Dictionary of evaluation metrics.
    """
    print(f"\nClassical ML (Random Forest on HOG)")

    # Standardise features
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train_hog)
    X_te = scaler.transform(X_test_hog)

    clf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    clf.fit(X_tr, y_train)
    elapsed = time.time() - t0

    preds = clf.predict(X_te)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=CLASSES)

    print(f"  Training time:  {elapsed:.1f}s")
    print(f"  Test accuracy:  {acc:.4f}")
    print(f"\n{report}")

    joblib.dump({"clf": clf, "scaler": scaler}, MODELS_DIR / "classical_model.pkl")
    _save_confusion_matrix(y_test, preds, "classical_confusion_matrix.png")

    return {"model": "classical", "accuracy": acc, "training_time_s": elapsed}


# 3. Deep Model

class ScribblNet(nn.Module):
    """Lightweight CNN for 28×28 grayscale sketch classification.

    Architecture:
        3 × (Conv2d → BatchNorm → ReLU → MaxPool)
        Dropout → FC(1152→256) → ReLU → Dropout → FC(256→num_classes)
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 28→14→7→3  ∴ feature map is 128×3×3 = 1152
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, 1, 28, 28), values in [0, 1].

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def make_dataloaders(
    X_raw: np.ndarray,
    y: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = DEEP_BATCH_SIZE,
    train_fraction: float = 1.0,
) -> tuple[DataLoader, DataLoader]:
    """Build PyTorch DataLoaders from raw pixel arrays.

    Pixel values are normalised to [0, 1].  Training set can be subsampled
    via train_fraction for the sensitivity experiment.

    Args:
        X_raw:          Training pixel array (N, 784), uint8.
        y:              Training labels.
        X_test_raw:     Test pixel array.
        y_test:         Test labels.
        batch_size:     Minibatch size.
        train_fraction: Fraction of training samples to use (0 < f ≤ 1).

    Returns:
        (train_loader, test_loader)
    """
    if train_fraction < 1.0:
        n = max(1, int(len(X_raw) * train_fraction))
        idx = np.random.default_rng(seed=7).permutation(len(X_raw))[:n]
        X_raw = X_raw[idx]
        y = y[idx]

    def _to_tensor(X: np.ndarray, labels: np.ndarray) -> TensorDataset:
        imgs = torch.from_numpy(X.astype(np.float32) / 255.0)
        imgs = imgs.view(-1, 1, IMG_SIZE, IMG_SIZE)
        return TensorDataset(imgs, torch.from_numpy(labels))

    train_ds = _to_tensor(X_raw, y)
    test_ds = _to_tensor(X_test_raw, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return average loss.

    Args:
        model:     ScribblNet instance.
        loader:    Training DataLoader.
        optimizer: Optimiser (Adam).
        criterion: Loss function (CrossEntropyLoss).
        device:    Torch device.

    Returns:
        Mean loss over all minibatches.
    """
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Evaluate model on a DataLoader.

    Args:
        model:  ScribblNet instance.
        loader: Evaluation DataLoader.
        device: Torch device.

    Returns:
        (accuracy, predictions_array)
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return accuracy_score(labels, preds), preds


def train_deep(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epochs: int = DEEP_EPOCHS,
    train_fraction: float = 1.0,
    save_model: bool = True,
) -> dict[str, Any]:
    """Train ScribblNet and evaluate on test set.

    Args:
        X_train_raw:    Raw training pixel array.
        X_test_raw:     Raw test pixel array.
        y_train:        Training labels.
        y_test:         Test labels.
        epochs:         Number of training epochs.
        train_fraction: Fraction of training data to use.
        save_model:     Whether to save weights to disk.

    Returns:
        Dictionary of evaluation metrics and training history.
    """
    print(f"\nDeep Model (ScribblNet, fraction={train_fraction:.0%})")
    device = get_device()
    print(f"  Device: {device}")

    train_loader, test_loader = make_dataloaders(
        X_train_raw, y_train, X_test_raw, y_test, train_fraction=train_fraction
    )

    model = ScribblNet(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=DEEP_LR, weight_decay=DEEP_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"loss": [], "val_acc": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, _ = evaluate(model, test_loader, device)
        scheduler.step()
        history["loss"].append(loss)
        history["val_acc"].append(acc)
        print(f"  epoch {epoch:02d}/{epochs}  loss={loss:.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            if save_model:
                torch.save(model.state_dict(), MODELS_DIR / "deep_model.pth")

    # Final evaluation with best weights
    if save_model:
        model.load_state_dict(torch.load(MODELS_DIR / "deep_model.pth", map_location=device))

    final_acc, final_preds = evaluate(model, test_loader, device)
    print(f"\n  Best test accuracy: {best_acc:.4f}")

    if save_model:
        report = classification_report(y_test, final_preds, target_names=CLASSES)
        print(f"\n{report}")
        _save_confusion_matrix(y_test, final_preds, "deep_confusion_matrix.png")
        _save_training_curves(history)

    return {"model": "deep", "accuracy": best_acc, "history": history}


# Experiment: Training Size Sensitivity

def run_experiment(
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_hog: np.ndarray,
    X_test_hog: np.ndarray,
) -> None:
    """Training set size sensitivity analysis.

    Sweeps over EXPERIMENT_FRACTIONS, training both the deep model and Random
    Forest at each fraction, then plots accuracy vs number of training samples.

    Motivation: Understanding how each model scales with data volume helps
    justify architectural choices and highlights when more data is beneficial.

    Args:
        X_train_raw:  Raw training pixels.
        X_test_raw:   Raw test pixels.
        y_train:      Training labels.
        y_test:       Test labels.
        X_train_hog:  HOG training features.
        X_test_hog:   HOG test features.
    """
    print(f"\nExperiment: Training Size Sensitivity")
    deep_accs, rf_accs, n_samples = [], [], []
    scaler = StandardScaler()
    X_test_scaled = scaler.fit_transform(X_test_hog)

    for frac in EXPERIMENT_FRACTIONS:
        n = int(len(X_train_raw) * frac)
        n_samples.append(n)
        print(f"\n  Fraction={frac:.0%}  (n={n})")

        # Deep model
        result = train_deep(
            X_train_raw, X_test_raw, y_train, y_test,
            epochs=EXPERIMENT_EPOCHS, train_fraction=frac, save_model=False,
        )
        deep_accs.append(result["accuracy"])

        # Random Forest
        idx = np.random.default_rng(seed=42).permutation(len(X_train_hog))[:n]
        X_tr = scaler.fit_transform(X_train_hog[idx])
        rf = RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=42
        )
        rf.fit(X_tr, y_train[idx])
        rf_pred = rf.predict(X_test_scaled)
        rf_accs.append(accuracy_score(y_test, rf_pred))
        print(f"  RF acc={rf_accs[-1]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_samples, deep_accs, marker="o", linestyle="solid", label="ScribblNet (CNN)", linewidth=2, markersize=7)
    ax.plot(n_samples, rf_accs, marker="s", linestyle="dashed", label="Random Forest (HOG)", linewidth=2, markersize=7)
    ax.set_xlabel("Training samples", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("Training Set Size Sensitivity", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = OUTPUTS_DIR / "experiment_sensitivity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\n  Saved experiment plot → {out_path}")

    results = {
        "fractions": EXPERIMENT_FRACTIONS,
        "n_samples": n_samples,
        "deep_accs": deep_accs,
        "rf_accs": rf_accs,
    }
    with open(OUTPUTS_DIR / "experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved experiment_results.json")


# Plotting Helpers

def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str,
) -> None:
    """Save a normalised confusion matrix heatmap.

    Args:
        y_true:   Ground truth labels.
        y_pred:   Predicted labels.
        filename: Output filename (saved under OUTPUTS_DIR).
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cmap="Blues",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(filename.replace("_", " ").replace(".png", "").title(), fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def _save_training_curves(history: dict[str, list[float]]) -> None:
    """Save loss and validation accuracy curves for the deep model.

    Args:
        history: Dict with keys 'loss' and 'val_acc', each a list of per epoch values.
    """
    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["loss"], color="steelblue", marker="o", linestyle="solid", markersize=5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("ScribblNet Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["val_acc"], color="seagreen", marker="o", linestyle="solid", markersize=5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Accuracy")
    ax2.set_title("ScribblNet Validation Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "deep_training_curves.png", dpi=150)
    plt.close(fig)
    print("  Saved deep_training_curves.png")


def _save_model_comparison(results: list[dict[str, Any]]) -> None:
    """Bar chart comparing test accuracy across all three models.

    Args:
        results: List of result dicts each containing 'model' and 'accuracy'.
    """
    names = [r["model"].capitalize() for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, accs, color=["#94a3b8", "#60a5fa", "#34d399"], width=0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Model Comparison")
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{acc:.3f}",
            ha="center",
            fontsize=12,
        )
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved model_comparison.png")


# Orchestrator

def train_all() -> None:
    """Train all three models, run the experiment, and save all artefacts."""
    X_train_raw, X_test_raw, y_train, y_test, X_train_hog, X_test_hog = (
        load_processed_data()
    )

    r_naive = train_naive(y_train, y_test)
    r_classical = train_classical(X_train_hog, X_test_hog, y_train, y_test)
    r_deep = train_deep(X_train_raw, X_test_raw, y_train, y_test)

    _save_model_comparison([r_naive, r_classical, r_deep])

    run_experiment(
        X_train_raw, X_test_raw, y_train, y_test, X_train_hog, X_test_hog
    )

    summary = {
        "naive_accuracy": r_naive["accuracy"],
        "classical_accuracy": r_classical["accuracy"],
        "deep_accuracy": r_deep["accuracy"],
    }
    with open(OUTPUTS_DIR / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete. Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    train_all()
