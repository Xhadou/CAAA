"""Download and extract RCAEval dataset.

Uses the `RCAEval <https://github.com/phamquiluan/RCAEval>`_ package's
built-in download utilities when available, falling back to direct Zenodo
URL downloads for RE1/RE2 if the package is not installed.

Install the RCAEval package from source for full support (RE1, RE2, **and RE3**)::

    pip install "RCAEval @ git+https://github.com/phamquiluan/RCAEval.git"
"""

import logging
import os
import zipfile
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

# Mapping from our system names to RCAEval's sub-dataset function suffixes.
# RCAEval uses per-system helpers like ``download_re1ob_dataset`` (OB = Online
# Boutique, SS = Sock Shop, TT = Train Ticket).
_SYSTEM_SUFFIX = {
    "online-boutique": "ob",
    "sock-shop": "ss",
    "train-ticket": "tt",
}

# Direct Zenodo URLs used as fallback when the RCAEval package is not
# installed.  Only RE1 and RE2 have known direct URLs.
RCAEVAL_URLS = {
    "RE1": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE1-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE1-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE1-train-ticket.zip",
    },
    "RE2": {
        "online-boutique": "https://zenodo.org/records/14590730/files/RE2-online-boutique.zip",
        "sock-shop": "https://zenodo.org/records/14590730/files/RE2-sock-shop.zip",
        "train-ticket": "https://zenodo.org/records/14590730/files/RE2-train-ticket.zip",
    },
}

# All datasets supported when the RCAEval package is installed.
AVAILABLE_DATASETS = ["RE1", "RE2", "RE3"]


def _has_rcaeval() -> bool:
    """Return True if the RCAEval.utility module is importable."""
    try:
        from RCAEval.utility import download_re1_dataset  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _download_via_rcaeval(
    dataset: str,
    systems: List[str],
    data_dir: str,
) -> None:
    """Download using the RCAEval package's built-in utilities.

    Each ``download_reN_dataset`` function downloads all three systems
    into ``<local_path>/REN/`` sub-directories.  Per-system helpers
    (``download_reNob_dataset`` etc.) are used when only a subset of
    systems is requested.
    """
    import importlib
    rcaeval_utility = importlib.import_module("RCAEval.utility")

    ds_lower = dataset.lower()  # e.g. "re1"

    for system in systems:
        suffix = _SYSTEM_SUFFIX.get(system)
        if suffix is None:
            print(f"Warning: unknown system '{system}' — skipping")
            continue

        fn_name = f"download_{ds_lower}{suffix}_dataset"
        fn = getattr(rcaeval_utility, fn_name, None)
        if fn is None:
            print(f"Warning: RCAEval has no function '{fn_name}' — skipping")
            continue

        dest = os.path.join(data_dir, dataset)
        print(f"Downloading {dataset}/{system} via RCAEval...")
        fn(local_path=dest)

    print("Download complete!")


# ------------------------------------------------------------------
# Fallback helpers (direct Zenodo downloads)
# ------------------------------------------------------------------

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


def _download_via_urls(
    dataset: str,
    systems: List[str],
    data_dir: str,
) -> None:
    """Fallback download using direct Zenodo URLs (RE1/RE2 only)."""
    urls = RCAEVAL_URLS.get(dataset, {})
    if not urls:
        raise ValueError(
            f"Dataset '{dataset}' cannot be downloaded without the RCAEval "
            f"package (only RE1/RE2 have direct URLs).  Install RCAEval from "
            f"source to download RE3:\n"
            f"  pip install \"RCAEval @ git+https://github.com/phamquiluan/RCAEval.git\""
        )

    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for system in systems:
        if system not in urls:
            print(f"Warning: {system} not found in {dataset}")
            continue

        url = urls[system]
        zip_path = data_path / f"{dataset}-{system}.zip"

        print(f"Downloading {dataset}/{system}...")
        download_file(url, zip_path)

        print(f"Extracting {dataset}/{system}...")
        extract_zip(zip_path, data_path / dataset / system)

    print("Download complete!")


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def download_rcaeval_dataset(
    dataset: str = "RE1",
    systems: Optional[List[str]] = None,
    data_dir: str = "data/raw",
) -> None:
    """Download an RCAEval dataset (RE1, RE2, or RE3).

    Uses the `RCAEval` package when available (supports all datasets
    including RE3).  Falls back to direct Zenodo URL downloads for
    RE1/RE2 if the package is not installed.

    Args:
        dataset: ``"RE1"``, ``"RE2"``, or ``"RE3"``.
        systems: List of systems to download.  Default: all three
            (online-boutique, sock-shop, train-ticket).
        data_dir: Directory to save data.

    Raises:
        ValueError: If *dataset* is not available via the current
            download method.
    """
    if systems is None:
        systems = ["online-boutique", "sock-shop", "train-ticket"]

    dataset = dataset.upper()
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available: {AVAILABLE_DATASETS}"
        )

    if _has_rcaeval():
        _download_via_rcaeval(dataset, systems, data_dir)
    else:
        logger.info(
            "RCAEval package not found — falling back to direct URL download. "
            "Install RCAEval from source for RE3 support: "
            "pip install \"RCAEval @ git+https://github.com/phamquiluan/RCAEval.git\""
        )
        _download_via_urls(dataset, systems, data_dir)


if __name__ == "__main__":
    download_rcaeval_dataset(dataset="RE1", systems=["online-boutique"])
