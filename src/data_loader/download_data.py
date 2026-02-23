"""Download and extract RCAEval dataset.

Downloads RCAEval benchmark datasets (RE1, RE2, RE3) directly from Zenodo.
When the `RCAEval <https://github.com/phamquiluan/RCAEval>`_ package is
installed, its built-in download utilities are used as an alternative.
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Mapping from our system names to RCAEval's abbreviated identifiers.
_SYSTEM_ABBREV = {
    "online-boutique": "OB",
    "sock-shop": "SS",
    "train-ticket": "TT",
}

# Direct Zenodo download URLs for all three dataset suites.
# Source: https://zenodo.org/records/14590730
RCAEVAL_URLS = {
    "RE1": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE1-OB.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE1-SS.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE1-TT.zip",
    },
    "RE2": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE2-OB.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE2-SS.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE2-TT.zip",
    },
    "RE3": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE3-OB.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE3-SS.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE3-TT.zip",
    },
}

AVAILABLE_DATASETS = list(RCAEVAL_URLS.keys())


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download file with progress bar.

    Args:
        url: URL to download.
        dest_path: Local path to save the downloaded file.
        chunk_size: Bytes per chunk for streaming download.
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests is required for downloading data. Install with: pip install requests")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=dest_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract zip file and remove the archive.

    Args:
        zip_path: Path to the zip file.
        extract_to: Directory to extract contents into.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)


def download_rcaeval_dataset(
    dataset: str = "RE1",
    systems: Optional[List[str]] = None,
    data_dir: str = "data/raw",
) -> None:
    """Download an RCAEval dataset (RE1, RE2, or RE3).

    Downloads the specified dataset suite from Zenodo for each requested
    system.  If the ``RCAEval`` package is installed (from source), its
    per-system helpers are used instead.

    Args:
        dataset: ``"RE1"``, ``"RE2"``, or ``"RE3"`` (case-insensitive).
        systems: List of systems to download.  Default: all three
            (online-boutique, sock-shop, train-ticket).
        data_dir: Directory to save data.

    Raises:
        ValueError: If *dataset* is not recognised.
    """
    if systems is None:
        systems = ["online-boutique", "sock-shop", "train-ticket"]

    dataset = dataset.upper()

    urls = RCAEVAL_URLS.get(dataset)
    if urls is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for system in systems:
        if system not in urls:
            print(f"Warning: {system} not found in {dataset}")
            continue

        url = urls[system]
        abbrev = _SYSTEM_ABBREV.get(system, system)
        zip_name = f"{dataset}-{abbrev}.zip"
        zip_path = data_path / zip_name

        print(f"Downloading {dataset}/{system}...")
        download_file(url, zip_path)

        print(f"Extracting {dataset}/{system}...")
        extract_zip(zip_path, data_path / dataset / system)

    print("Download complete!")


if __name__ == "__main__":
    download_rcaeval_dataset(dataset="RE1", systems=["online-boutique"])
