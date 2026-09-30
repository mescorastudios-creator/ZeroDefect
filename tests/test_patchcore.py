import pytest

from inspection.patchcore import Inspector, _best_threshold, auroc
from ml.datasets import demo
from ml.datasets.common import load_manifests


def test_threshold_and_auroc():
    import numpy as np

    assert _best_threshold(np.array([1.0, 1.1, 1.2]), np.array([2.0, 2.5])) == pytest.approx(1.6)
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert auroc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5


def test_patchcore_finds_defects_on_held_out_parts(tmp_path, monkeypatch):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(tmp_path))
    demo.generate(
        part_types=("u_block",), size=128, train_good=10, train_per_defect=2, test_good=4, test_per_defect=2
    )
    inspector = Inspector.load_or_fit("demo", tmp_path / "models")
    metrics = inspector.models["u_block"].metrics
    assert metrics["parts"] == 4 + 2 * len(demo.DEFECTS)
    assert metrics["image_auroc"] > 0.9
    assert metrics["accuracy"] > 0.8

    test = load_manifests(["demo"]).query("split == 'test'")
    import cv2

    from ml.datasets.common import resolve

    bad = test[test.zd_class == "missing_material"].groupby("sample_id").image_path.apply(list).iloc[0]
    res = inspector.inspect_part("u_block", [cv2.imread(str(resolve(p))) for p in bad])
    assert res["defect"] and res["views"][0]["boxes"]
