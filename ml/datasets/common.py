"""Unified dataset format shared by every source converter.

Each converted ``<source>/<category>`` gets a folder under ``data/processed``:

* ``manifest.jsonl``: one :class:`Sample` per image (split, label, taxonomy class,
  physical sample id and camera view).
* ``coco.json``: COCO instances whose category ids are the taxonomy class ids.
  Instances come from the source boxes or from connected regions of the masks.

Image and mask paths are POSIX paths relative to the data root, so a data folder
can be moved or mounted elsewhere unchanged.
"""

import io
import json
import logging
import shutil
import tarfile
import time
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import pandas as pd
import requests
import urllib3
from PIL import Image
from tqdm import tqdm

from inspection.paths import data_root
from inspection.taxonomy import load_taxonomy

log = logging.getLogger(__name__)

GOOD, DEFECT = "good", "defect"
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    image_id: str  # unique: "<source>/<category>/<...>"
    source: str
    category: str  # part type, e.g. plastic_nut
    split: str  # train | val | test
    image_path: str  # relative to the data root
    label: str  # good | defect
    zd_class: str | None = None  # taxonomy class key, None for good images
    zd_subtype: str | None = None
    source_label: str | None = None  # original dataset label, kept for traceability
    mask_path: str | None = None
    sample_id: str | None = None  # physical part; groups the views of one part
    view: str | None = None  # camera view, e.g. C1
    width: int | None = None
    height: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Instance:
    """One annotated defect region, in pixel coordinates of its image."""

    zd_class: str
    bbox: list[float]  # x, y, w, h
    area: float
    segmentation: list[list[float]] = field(default_factory=list)
    subtype: str | None = None
    source_label: str | None = None


def raw_dir(source: str) -> Path:
    return data_root() / "raw" / source


def processed_dir(source: str, category: str) -> Path:
    return data_root() / "processed" / source / category


def rel(path: Path) -> str:
    return path.resolve().relative_to(data_root().resolve()).as_posix()


def resolve(relpath: str) -> Path:
    return data_root() / relpath


# --------------------------------------------------------------------------- masks


def read_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Binary uint8 mask (0/1), resized with nearest neighbour to ``size`` (w, h) if given."""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    if size is not None and (mask.shape[1], mask.shape[0]) != size:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def instances_from_mask(
    mask: np.ndarray, zd_class: str, merge_px: int = 5, min_area: int = 4, **attrs
) -> list[Instance]:
    """Split a binary mask into defect instances.

    Regions closer than ``merge_px`` pixels are grouped into one instance, so a
    scratch that the annotator drew in several strokes stays one defect.
    """
    fg = (mask > 0).astype(np.uint8)
    grouped = cv2.dilate(fg, np.ones((2 * merge_px + 1,) * 2, np.uint8)) if merge_px else fg
    n, labels = cv2.connectedComponents(grouped)
    out = []
    for k in range(1, n):
        comp = ((labels == k) & (fg > 0)).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(comp)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        seg = [c.reshape(-1).astype(float).tolist() for c in contours if len(c) >= 3]
        if not seg:  # a line or dot: fall back to the box outline
            seg = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        out.append(Instance(zd_class, [x, y, w, h], area, seg, **attrs))
    return out


# --------------------------------------------------------------------------- writing


def write_dataset(
    out_dir: Path, samples: list[Sample], instances: dict[str, list[Instance]] | None = None
) -> Path:
    """Write ``manifest.jsonl`` + ``coco.json`` for one source/category.

    Image sizes are filled in from the files. Defect samples with a mask and no
    explicit instances get instances extracted from the mask.
    """
    tax = load_taxonomy()
    instances = dict(instances or {})
    out_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i, s in enumerate(tqdm(samples, desc=f"write {out_dir.name}", leave=False), start=1):
        if s.width is None or s.height is None:
            with Image.open(resolve(s.image_path)) as im:
                s.width, s.height = im.size
        if s.image_id not in instances and s.mask_path and s.zd_class:
            mask = read_mask(resolve(s.mask_path), (s.width, s.height))
            instances[s.image_id] = instances_from_mask(
                mask, s.zd_class, subtype=s.zd_subtype, source_label=s.source_label
            )
        images.append(
            {
                "id": i,
                "file_name": s.image_path,
                "width": s.width,
                "height": s.height,
                "zd_image_id": s.image_id,
                "split": s.split,
                "label": s.label,
                "sample_id": s.sample_id,
                "view": s.view,
            }
        )
        for inst in instances.get(s.image_id, []):
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": i,
                    "category_id": tax.get(inst.zd_class).id,
                    "bbox": inst.bbox,
                    "area": inst.area,
                    "segmentation": inst.segmentation,
                    "iscrowd": 0,
                    "subtype": inst.subtype,
                    "source_label": inst.source_label,
                }
            )

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")
    coco = {
        "info": {"description": "ZeroDefect unified format", "taxonomy_version": tax.version},
        "images": images,
        "annotations": annotations,
        "categories": tax.coco_categories(),
    }
    (out_dir / "coco.json").write_text(json.dumps(coco), encoding="utf-8")
    log.info("%s: %d images, %d instances", out_dir, len(images), len(annotations))
    return out_dir


def read_manifest(path: Path) -> list[Sample]:
    with open(path, encoding="utf-8") as f:
        return [Sample(**json.loads(line)) for line in f if line.strip()]


def load_manifests(sources: Iterable[str] | None = None, root: Path | None = None) -> pd.DataFrame:
    """All converted samples as one DataFrame (one row per image)."""
    root = root or data_root() / "processed"
    wanted = set(sources) if sources else None
    rows = []
    for path in sorted(root.glob("*/*/manifest.jsonl")):
        if wanted is None or path.parent.parent.name in wanted:
            rows.extend(asdict(s) for s in read_manifest(path))
    return pd.DataFrame(rows, columns=[f.name for f in Sample.__dataclass_fields__.values()])


# --------------------------------------------------------------------------- downloads


class HTTPStream(io.RawIOBase):
    """Read-only HTTP body that resumes with Range requests after a dropped connection.

    Lets us pipe multi-GB tar archives through ``tarfile`` without storing them.
    """

    def __init__(self, url: str, headers: dict | None = None, retries: int = 8, progress: bool = True):
        self.url, self.headers, self.retries = url, headers or {}, retries
        self.pos, self.size = 0, None
        self._resp = self._bar = None
        self._open()
        self._bar = tqdm(
            total=self.size, unit="B", unit_scale=True, desc=url.rsplit("/", 1)[-1][:40], disable=not progress
        )

    def _open(self):
        headers = dict(self.headers)
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        resp = requests.get(self.url, headers=headers, stream=True, timeout=60)
        resp.raise_for_status()
        if self.pos and resp.status_code != 206:
            raise OSError(f"{self.url}: server ignored the Range header, cannot resume")
        if self.size is None and "Content-Length" in resp.headers:
            self.size = int(resp.headers["Content-Length"])
        self._resp = resp

    def readable(self):
        return True

    def readinto(self, b) -> int:
        for attempt in range(self.retries + 1):
            try:
                if self._resp is None:
                    self._open()
                data = self._resp.raw.read(len(b))
                if not data and self.size is not None and self.pos < self.size:
                    raise OSError("connection closed early")
                break
            except (OSError, requests.RequestException, urllib3.exceptions.HTTPError) as e:
                if attempt == self.retries:
                    raise
                log.warning("download interrupted at %d bytes (%s); resuming", self.pos, e)
                if self._resp is not None:
                    self._resp.close()
                    self._resp = None
                time.sleep(min(2**attempt, 30))
        n = len(data)
        b[:n] = data
        self.pos += n
        self._bar.update(n)
        return n

    def close(self):
        if self._resp is not None:
            self._resp.close()
        if self._bar is not None:
            self._bar.close()
        super().close()


def download_file(url: str, dest: Path, headers: dict | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with HTTPStream(url, headers) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    tmp.replace(dest)
    return dest


def _class_path(name: str, classes: set[str], max_depth: int = 2) -> str | None:
    """Archive member path re-rooted at the first component that is a wanted class."""
    parts = PurePosixPath(name.lstrip("./")).parts
    for k, part in enumerate(parts[:max_depth]):
        if part in classes:
            return "/".join(parts[k:])
    return None


def extract_classes_from_tar(url: str, dest: Path, classes: Iterable[str]) -> None:
    """Stream a remote tar archive and extract only the given class folders into ``dest``.

    Tar archives of a directory tree store each folder contiguously, so the
    download stops as soon as every wanted class folder has been passed.
    """
    wanted = set(classes)
    done: set[str] = set()
    current = None
    dest.mkdir(parents=True, exist_ok=True)
    with (
        io.BufferedReader(HTTPStream(url), buffer_size=1 << 20) as stream,
        tarfile.open(fileobj=stream, mode="r|*") as tar,
    ):
        for member in tar:
            path = _class_path(member.name, wanted)
            cls = path.split("/", 1)[0] if path else None
            if current is not None and cls != current:
                done.add(current)
            current = cls
            if done >= wanted:
                break
            if path and (member.isfile() or member.isdir()):
                member.name = path
                tar.extract(member, dest, filter="data")
    missing = wanted - done - ({current} if current else set())
    if missing:
        raise FileNotFoundError(f"classes not found in archive {url}: {sorted(missing)}")


def extract_zip(path: Path, dest: Path, keep: Callable[[str], bool] = lambda name: True) -> None:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if keep(info.filename):
                zf.extract(info, dest)
