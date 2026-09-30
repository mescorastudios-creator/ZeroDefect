from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def data_root(tmp_path, monkeypatch) -> Path:
    """Point ZERODEFECT_DATA_ROOT at an empty temp folder."""
    root = tmp_path / "data"
    root.mkdir()
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(root))
    return root


def write_image(path: Path, size=(32, 32), value=128) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((size[1], size[0], 3), value, np.uint8))
    return path


def write_mask(path: Path, boxes, size=(32, 32)) -> Path:
    """Mask with filled rectangles ``(x0, y0, x1, y1)`` (inclusive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = np.zeros((size[1], size[0]), np.uint8)
    for x0, y0, x1, y1 in boxes:
        m[y0 : y1 + 1, x0 : x1 + 1] = 255
    cv2.imwrite(str(path), m)
    return path


@pytest.fixture(scope="session")
def demo_data(tmp_path_factory) -> Path:
    """A small procedural demo dataset shared by simulator and camera tests."""
    root = tmp_path_factory.mktemp("demo") / "data"
    mp = pytest.MonkeyPatch()
    mp.setenv("ZERODEFECT_DATA_ROOT", str(root))
    from ml.datasets import demo

    demo.generate(size=64, train_good=2, train_per_defect=1, test_good=3, test_per_defect=1)
    mp.undo()
    return root
