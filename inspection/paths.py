"""Repository locations shared by every package."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def data_root() -> Path:
    """Root for datasets and simulator runs; override with ZERODEFECT_DATA_ROOT."""
    return Path(os.environ.get("ZERODEFECT_DATA_ROOT", REPO_ROOT / "data"))
