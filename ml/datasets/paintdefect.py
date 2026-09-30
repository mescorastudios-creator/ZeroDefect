"""PaintDefect (Roboflow Universe): supervised defect boxes on painted plastic car parts.

Downloaded as a Roboflow COCO export (``data/raw/paintdefect/v<N>``)::

    {train,valid,test}/_annotations.coco.json + images

Needs ``ROBOFLOW_API_KEY``. License CC BY 4.0 (credit the authors):
https://universe.roboflow.com/ai-klghd/paintdefect-8h4s4
"""

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

import requests

from inspection.taxonomy import load_taxonomy
from ml.datasets.common import (
    DEFECT,
    GOOD,
    Instance,
    Sample,
    download_file,
    extract_zip,
    processed_dir,
    raw_dir,
    rel,
    write_dataset,
)

log = logging.getLogger(__name__)

SOURCE = "paintdefect"
CATEGORY = "painted_panel"
WORKSPACE, PROJECT = "ai-klghd", "paintdefect-8h4s4"
API = "https://api.roboflow.com"
_SPLITS = {"train": "train", "valid": "val", "test": "test"}


def _api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise PermissionError("ROBOFLOW_API_KEY is not set (see docs/SETUP.md, step 2)")
    return key


def _get(url: str, key: str) -> dict:
    resp = requests.get(url, params={"api_key": key}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def latest_version(key: str) -> int:
    versions = _get(f"{API}/{WORKSPACE}/{PROJECT}", key)["versions"]
    return max(int(v["id"].rsplit("/", 1)[-1]) for v in versions)


def download(version: int | None = None) -> Path:
    key = _api_key()
    version = version or latest_version(key)
    target = raw_dir(SOURCE) / f"v{version}"
    if any(target.glob("*/_annotations.coco.json")):
        return target
    # Roboflow prepares the export on first request; poll until the link is ready.
    for _ in range(60):
        info = _get(f"{API}/{WORKSPACE}/{PROJECT}/{version}/coco", key)
        link = info.get("export", {}).get("link")
        if link:
            break
        time.sleep(5)
    else:
        raise TimeoutError("Roboflow export was not ready after 5 minutes")
    archive = download_file(link, raw_dir(SOURCE) / f"_downloads/v{version}.zip")
    extract_zip(archive, target)
    archive.unlink()
    return target


def convert(version: int | None = None) -> list[Path]:
    root = raw_dir(SOURCE)
    if version is None:
        versions = sorted(root.glob("v*"), key=lambda p: int(p.name[1:]))
        if not versions:
            raise FileNotFoundError(f"no export in {root}; run the paintdefect download first")
        root = versions[-1]
    else:
        root = root / f"v{version}"
    tax = load_taxonomy()
    samples, instances = [], {}
    for folder, split in _SPLITS.items():
        ann_file = root / folder / "_annotations.coco.json"
        if not ann_file.exists():
            continue
        coco = json.loads(ann_file.read_text(encoding="utf-8"))
        names = {c["id"]: c["name"] for c in coco["categories"]}
        by_image: dict[int, list[dict]] = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        # Roboflow adds an unused "superclass" category; only map labels in use.
        used = {names[a["category_id"]] for a in coco["annotations"]}
        unmapped = sorted(n for n in used if n.lower() not in tax.source_labels(SOURCE))
        if unmapped:
            raise KeyError(f"PaintDefect labels not mapped in configs/taxonomy.yaml: {unmapped}")

        for im in coco["images"]:
            img = root / folder / im["file_name"]
            image_id = f"{SOURCE}/{CATEGORY}/{split}/{Path(im['file_name']).stem}"
            # Roboflow names augmented copies "<original>.rf.<hash>.jpg"; the copies are one physical part.
            original = im["file_name"].split(".rf.", 1)[0]
            insts = []
            for a in by_image.get(im["id"], []):
                label = names[a["category_id"]]
                m = tax.map_label(SOURCE, label)
                insts.append(
                    Instance(
                        m.zd_class,
                        a["bbox"],
                        a.get("area") or a["bbox"][2] * a["bbox"][3],
                        a.get("segmentation") or [],
                        m.subtype,
                        label,
                    )
                )
            # Image-level class: the most frequent defect class in the image.
            main = None
            if insts:
                top_class = Counter(i.zd_class for i in insts).most_common(1)[0][0]
                main = next(i for i in insts if i.zd_class == top_class)
            samples.append(
                Sample(
                    image_id=image_id,
                    source=SOURCE,
                    category=CATEGORY,
                    split=split,
                    image_path=rel(img),
                    label=DEFECT if main else GOOD,
                    zd_class=main and main.zd_class,
                    zd_subtype=main and main.subtype,
                    source_label=main and main.source_label,
                    sample_id=f"{SOURCE}/{CATEGORY}/{split}/{original}",
                    width=im.get("width"),
                    height=im.get("height"),
                )
            )
            instances[image_id] = insts
    if not samples:
        raise FileNotFoundError(f"no _annotations.coco.json under {root}")
    return [write_dataset(processed_dir(SOURCE, CATEGORY), samples, instances)]
