"""YOLO dataset from the unified format (``data/processed``).

Output in ``data/yolo/<name>``::

    images/<split>/<source>__<category>__<stem>.jpg   hard links to the originals (copies if that fails)
    labels/<split>/<same stem>.txt                     one line per defect, normalised coordinates
    data.yaml                                          Ultralytics dataset file (paths relative to it)
    index.csv                                          one row per image: split, part, view, label, boxes

YOLO class index = taxonomy id - 1, so classes keep their index when the taxonomy grows.
"""

import json
import logging
import os
import shutil
import zlib
from pathlib import Path

import pandas as pd
import yaml

from inspection.paths import data_root
from inspection.taxonomy import load_taxonomy
from ml.datasets.common import DEFECT, read_manifest, resolve

log = logging.getLogger(__name__)

TASKS = ("detect", "segment")
SPLITS = ("train", "val", "test")
DEFECT_PART_SPLIT = (
    0.70,
    0.15,
)  # train / val share of defective parts in anomaly-style sources; rest is test
GOOD_VAL_SHARE = 0.15  # share of the official good training parts moved to val in anomaly-style sources
BACKGROUND_RATIO = 0.25  # images without defects per image with defects, in train and val
DROP_CLASSES = ("unknown",)  # anomaly-model fallback, not a detector class


def yolo_dir(name: str) -> Path:
    return data_root() / "yolo" / name


def class_names() -> dict[int, str]:
    return {c.id - 1: c.key for c in load_taxonomy().classes}


def _bucket(key: str) -> float:
    """Deterministic number in [0, 1) for a key."""
    return zlib.crc32(key.encode()) / 2**32


def detector_splits(df: pd.DataFrame) -> pd.Series:
    """Detector split per image, decided per physical part so the views of one part stay together.

    Supervised sources, whose train split already has defects (PaintDefect, demo), keep their split.
    Anomaly-style sources (Real-IAD, MVTec) put every defective part in test. Their defective parts
    are re-split 70/15/15; good parts in the official test stay in test, and 15% of the official
    good training parts move to val. The detector test set therefore stays inside the official test
    set, which the anomaly model never trains on, so both models can be scored on the same parts.
    """
    part = df["sample_id"].fillna(df["image_id"])
    out = df["split"].copy()
    for _, g in df.groupby(["source", "category"]):
        if ((g["split"] == "train") & (g["label"] == DEFECT)).any():
            continue
        defective = set(part[g.index][g["label"] == DEFECT])
        for idx, pid in part[g.index].items():
            b = _bucket(pid)
            if pid in defective:
                out[idx] = (
                    "train" if b < DEFECT_PART_SPLIT[0] else "val" if b < sum(DEFECT_PART_SPLIT) else "test"
                )
            elif df.at[idx, "split"] == "train":
                out[idx] = "val" if b < GOOD_VAL_SHARE else "train"
    return out


def _load(sources, categories) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    """Samples of the wanted sources/categories plus their COCO annotations keyed by image_id."""
    rows, anns = [], {}
    root = data_root() / "processed"
    for src in sources:
        folders = sorted(p.parent for p in (root / src).glob("*/manifest.jsonl"))
        if not folders:
            raise FileNotFoundError(f"no converted {src} data under {root / src}; run ml.datasets convert")
        for folder in folders:
            if categories and folder.name not in categories:
                continue
            rows.extend(vars(s) for s in read_manifest(folder / "manifest.jsonl"))
            coco = json.loads((folder / "coco.json").read_text(encoding="utf-8"))
            ids = {im["id"]: im["zd_image_id"] for im in coco["images"]}
            for a in coco["annotations"]:
                anns.setdefault(ids[a["image_id"]], []).append(a)
    if not rows:
        raise FileNotFoundError(
            f"no samples for sources {list(sources)} and categories {list(categories or [])}"
        )
    return pd.DataFrame(rows), anns


def _label_lines(anns: list[dict], w: int, h: int, task: str) -> list[str]:
    lines = []
    for a in anns:
        cls = a["category_id"] - 1
        if task == "detect":
            x, y, bw, bh = a["bbox"]
            x0, y0 = max(0.0, x / w), max(0.0, y / h)
            x1, y1 = min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)
            if x1 > x0 and y1 > y0:
                lines.append(f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        else:
            if not a.get("segmentation"):
                raise ValueError(
                    f"image annotation {a['id']} has no polygon; use --task detect for box-only data"
                )
            # A defect drawn in several strokes has several contours: one polygon line each.
            for poly in a["segmentation"]:
                xy = [min(1.0, max(0.0, v / (w if i % 2 == 0 else h))) for i, v in enumerate(poly)]
                lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in xy))
    return lines


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(
    name: str,
    sources,
    categories=None,
    task: str = "detect",
    background_ratio: float = BACKGROUND_RATIO,
    drop_classes=DROP_CLASSES,
    seed: int = 0,
) -> Path:
    """Write a YOLO dataset; returns the path of its ``data.yaml``.

    Train and val keep every image with defects plus ``background_ratio`` images without defects
    per defect image. Test keeps every image of its parts, so line-level metrics see whole parts.
    Images whose defect has no box, or only boxes of ``drop_classes``, are left out: as background
    they would teach the detector that a real defect is normal surface.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    tax = load_taxonomy()
    drop_ids = {tax.get(k).id for k in drop_classes}
    df, anns = _load(sources, categories)
    df["det_split"] = detector_splits(df)
    df["n_boxes"] = [len(anns.get(i, [])) for i in df["image_id"]]
    dropped = [i for i in df["image_id"] if any(a["category_id"] in drop_ids for a in anns.get(i, []))]
    unboxed = df.index[(df["label"] == DEFECT) & (df["n_boxes"] == 0)]
    df = df.drop(index=unboxed)
    df = df[~df["image_id"].isin(dropped)]
    if len(unboxed) or dropped:
        log.info(
            "left out %d defect images without boxes and %d with %s", len(unboxed), len(dropped), drop_classes
        )

    keep = []
    for (split, _, _), g in df.groupby(["det_split", "source", "category"]):
        has_boxes = g["n_boxes"] > 0
        keep.append(g[has_boxes])
        background = g[~has_boxes]
        if split != "test":
            n = round(background_ratio * has_boxes.sum())
            order = background["image_id"].map(lambda i: _bucket(f"{seed}:{i}")).sort_values().index
            background = background.loc[order[:n]]
        keep.append(background)
    df = pd.concat(keep).sort_values(["det_split", "image_id"])

    out = yolo_dir(name)
    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    files = []
    for r in df.itertuples():
        src = resolve(r.image_path)
        stem = r.image_id.replace("/", "__")
        _place(src, out / "images" / r.det_split / f"{stem}{src.suffix.lower()}")
        lines = _label_lines(anns.get(r.image_id, []), r.width, r.height, task)
        (out / "labels" / r.det_split / f"{stem}.txt").write_text("".join(f"{s}\n" for s in lines))
        files.append(f"images/{r.det_split}/{stem}{src.suffix.lower()}")

    index = pd.DataFrame(
        {
            "file": files,
            "split": df["det_split"].to_numpy(),
            "source_split": df["split"].to_numpy(),
            "part": df["sample_id"].fillna(df["image_id"]).to_numpy(),
            **{
                c: df[c].to_numpy()
                for c in ("image_id", "view", "source", "category", "label", "zd_class", "n_boxes")
            },
        }
    )
    index.to_csv(out / "index.csv", index=False)

    data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names(),
    }
    yaml_path = out / "data.yaml"
    header = (
        f"# ZeroDefect {task} dataset from {', '.join(sources)}; built by python -m ml.detector build.\n"
        "# Class index = taxonomy id - 1 (configs/taxonomy.yaml).\n"
    )
    yaml_path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    log.info("%s\n%s", yaml_path, summary(index).to_string())
    return yaml_path


def summary(index: pd.DataFrame) -> pd.DataFrame:
    """Images, parts, defect images and boxes per split."""
    g = index.groupby("split")
    table = pd.DataFrame(
        {
            "images": g.size(),
            "parts": g["part"].nunique(),
            "defect_images": g["n_boxes"].apply(lambda s: int((s > 0).sum())),
            "background": g["n_boxes"].apply(lambda s: int((s == 0).sum())),
            "boxes": g["n_boxes"].sum(),
        }
    )
    return table.reindex([s for s in SPLITS if s in table.index])
