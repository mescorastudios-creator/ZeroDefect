import pandas as pd
import pytest
import yaml
from conftest import write_image, write_mask

from inspection.paths import data_root
from ml.datasets.common import Instance, Sample, processed_dir, read_mask, rel, write_dataset
from ml.detector import dataset
from ml.detector.train import line_metrics

N_DEFECT_PARTS = 30


def _anomaly_source():
    """Real-IAD-like: good parts in train and test, defective parts (2 views) only in test."""
    root = data_root() / "raw/fake"
    samples = []
    for pid in range(40):
        defective = pid < N_DEFECT_PARTS
        split = "test" if defective or pid % 2 else "train"
        for view in ("C1", "C2"):
            img = write_image(root / f"p{pid:02d}_{view}.jpg")
            mask = None
            # View C2 of defective part 0 shows nothing: labelled good. Part 1's C2 mask is empty.
            label = "defect" if defective and not (pid == 0 and view == "C2") else "good"
            if label == "defect":
                mask = write_mask(
                    root / f"p{pid:02d}_{view}.png", [] if pid == 1 and view == "C2" else [(4, 4, 9, 9)]
                )
            samples.append(
                Sample(
                    image_id=f"fake/nut/p{pid:02d}_{view}",
                    source="fake",
                    category="nut",
                    split=split,
                    image_path=rel(img),
                    label=label,
                    zd_class="pit_void" if label == "defect" else None,
                    mask_path=rel(mask) if mask else None,
                    sample_id=f"fake/nut/p{pid:02d}",
                    view=view,
                )
            )
    write_dataset(processed_dir("fake", "nut"), samples)
    assert read_mask(data_root() / samples[3].mask_path).sum() == 0  # the empty mask of part 1, C2


def _box_source():
    """PaintDefect-like: its own train/val/test split with defects in train, boxes only."""
    root = data_root() / "raw/boxes"
    samples, instances = [], {}
    for i, (split, cls) in enumerate(
        [("train", "scratch"), ("train", None), ("train", None), ("val", "unknown"), ("test", "scratch")]
    ):
        img = write_image(root / f"b{i}.jpg", size=(64, 32))
        image_id = f"boxes/panel/b{i}"
        samples.append(
            Sample(
                image_id=image_id,
                source="boxes",
                category="panel",
                split=split,
                image_path=rel(img),
                label="defect" if cls else "good",
                zd_class=cls,
                sample_id=image_id,
            )
        )
        if cls:
            instances[image_id] = [Instance(cls, [16, 8, 32, 16], 512)]
    write_dataset(processed_dir("boxes", "panel"), samples, instances)


@pytest.fixture
def built(data_root):
    _anomaly_source()
    _box_source()
    yaml_path = dataset.build("t", ["fake", "boxes"], background_ratio=0.5)
    return yaml_path, pd.read_csv(yaml_path.parent / "index.csv")


def test_data_yaml_uses_taxonomy_order(built):
    yaml_path, _ = built
    data = yaml.safe_load(yaml_path.read_text())
    assert (
        data["names"][0] == "scratch"
        and data["names"][3] == "pit_void"
        and data["names"][8] == "paint_finish"
    )
    assert (data["train"], data["val"], data["test"]) == ("images/train", "images/val", "images/test")
    assert "path" not in data  # relative to the yaml, so the folder can be moved (e.g. to Colab)


def test_parts_never_straddle_splits(built):
    _, index = built
    assert (index.groupby("part")["split"].nunique() == 1).all()
    fake = index[index.source == "fake"]
    # Every split gets defective parts; test stays inside the official test split.
    assert set(fake[fake.n_boxes > 0].split) == {"train", "val", "test"}
    assert (fake[fake.split == "test"].source_split == "test").all()


def test_supervised_source_keeps_its_split(built):
    _, index = built
    boxes = index[index.source == "boxes"].set_index("image_id")
    assert boxes.loc["boxes/panel/b0", "split"] == "train"
    assert boxes.loc["boxes/panel/b4", "split"] == "test"


def test_unboxed_and_dropped_defects_are_left_out(built):
    _, index = built
    ids = set(index.image_id)
    assert "fake/nut/p01_C2" not in ids  # defect label but empty mask: would teach "defect = normal"
    assert "boxes/panel/b3" not in ids  # only an "unknown" box


def test_background_ratio_in_train_and_whole_parts_in_test(built):
    _, index = built
    train = index[(index.split == "train") & (index.source == "boxes")]
    assert (train.n_boxes > 0).sum() == 1 and (train.n_boxes == 0).sum() == 0  # round(0.5 * 1) = 0
    test_parts = index[(index.split == "test") & (index.source == "fake")]
    good_test_parts = test_parts[test_parts.n_boxes == 0].part.unique()
    assert len(good_test_parts) > 0
    assert (test_parts.groupby("part").size() == 2).all()


def test_labels_are_normalised_boxes(built):
    yaml_path, index = built
    root = yaml_path.parent
    row = index[(index.source == "fake") & (index.n_boxes > 0)].iloc[0]
    label = (root / row.file.replace("images/", "labels/", 1)).with_suffix(".txt").read_text().split()
    # 6x6 px box at (4, 4) in a 32x32 image; pit_void is taxonomy id 4 -> class 3
    assert label == ["3", "0.218750", "0.218750", "0.187500", "0.187500"]
    b0 = (root / "labels/train/boxes__panel__b0.txt").read_text().split()
    assert b0 == ["0", "0.500000", "0.500000", "0.500000", "0.500000"]
    assert all((root / f).exists() for f in index.file)


def test_segment_needs_polygons(data_root):
    _box_source()
    with pytest.raises(ValueError, match="polygon"):
        dataset.build("seg", ["boxes"], task="segment")


def test_segment_from_masks(data_root):
    _anomaly_source()
    root = dataset.build("seg", ["fake"], task="segment").parent
    line = next(t for t in (root / "labels/train").glob("*.txt") if t.read_text()).read_text().split()
    assert line[0] == "3" and len(line) >= 7 and len(line) % 2 == 1


def test_line_metrics_views_and_single_view_parts():
    pred = pd.DataFrame(
        {
            "file": list("abcdefg"),
            "part": ["good", "good", "bad", "bad", "miss", "miss", "panel"],
            "n_boxes": [0, 0, 1, 0, 1, 0, 1],
            "max_conf": [0.3, 0.1, 0.9, 0.4, 0.2, 0.1, 0.8],
        }
    )
    m = line_metrics(pred, conf=0.25)
    assert m["image"]["false_reject_rate"] == pytest.approx(2 / 4)  # a, d
    assert m["image"]["escape_rate"] == pytest.approx(1 / 3)  # e
    assert 0.5 < m["image"]["auroc"] <= 1
    k1, k2 = m["part"]["min_views_1"], m["part"]["min_views_2"]
    assert (k1["good"], k1["defective"]) == (1, 3)
    assert k1["false_reject_rate"] == 1.0 and k1["escape_rate"] == pytest.approx(1 / 3)  # miss
    # Two views needed: "good" (1 flagged view) now passes, "bad" (2) fails, the 1-view panel still fails.
    assert k2["false_reject_rate"] == 0.0 and k2["escape_rate"] == pytest.approx(1 / 3)
