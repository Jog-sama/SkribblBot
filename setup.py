"""
setup.py – Orchestrates the full ScribblBot pipeline:
    1. Download Quick Draw data (make_dataset.py)
    2. Build features           (build_features.py)
    3. Train all models         (model.py)

Usage:
    python setup.py            # run full pipeline
    python setup.py --skip_download  # skip if data already downloaded
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.make_dataset import download_all
from scripts.build_features import build_all
from scripts.model import train_all


def run(skip_download: bool = False) -> None:
    """Execute the complete data and training pipeline.

    Args:
        skip_download: If True, skip the dataset download step.
                       Useful when raw .npy files are already present.
    """
    print("ScribblBot setup pipeline starting")

    if not skip_download:
        print("\n[1/3] Downloading dataset ...")
        download_all()
    else:
        print("\n[1/3] Skipping download (--skip_download)")

    print("\n[2/3] Building features ...")
    build_all()

    print("\n[3/3] Training models ...")
    train_all()

    print("\nSetup complete. Run the app with:")
    print("  python app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScribblBot full pipeline setup")
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip dataset download (use if .npy files already exist in data/raw/)",
    )
    args = parser.parse_args()
    run(skip_download=args.skip_download)
