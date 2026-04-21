"""
make_dataset.py – Download Quick Draw .npy files for all configured classes.

Usage:
    python scripts/make_dataset.py
"""

import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CLASSES, RAW_DIR, QUICKDRAW_URL


def download_class(cls: str, dest_dir: Path, force: bool = False) -> Path:
    """Download the numpy bitmap file for a single Quick Draw class.

    Args:
        cls:      Class name matching a Quick Draw category (e.g. 'cat').
        dest_dir: Directory to write the .npy file into.
        force:    Redownload even if the file already exists.

    Returns:
        Path to the downloaded file.
    """
    url = QUICKDRAW_URL.format(cls=cls.replace(" ", "%20"))
    dest = dest_dir / f"{cls}.npy"
    if dest.exists() and not force:
        print(f"  [skip] {cls}.npy already exists")
        return dest

    print(f"  [down] {cls}.npy  ->  {url}")

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        pct = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        print(f"\r         {pct:3d}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print()
    return dest


def download_all(force: bool = False) -> None:
    """Download .npy files for every class listed in config.CLASSES.

    Args:
        force: Redownload files that already exist on disk.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(CLASSES)} classes to {RAW_DIR} …\n")
    for cls in CLASSES:
        download_class(cls, RAW_DIR, force=force)
    print("\nAll downloads complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Quick Draw dataset")
    parser.add_argument("--force", action="store_true", help="Redownload existing files")
    args = parser.parse_args()
    download_all(force=args.force)
