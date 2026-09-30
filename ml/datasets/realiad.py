"""Real-IAD (CVPR 2024): the main dataset. Real plastic/rubber parts, 5 camera views each.

Layout after download (``data/raw/realiad``)::

    realiad_<res>/<category>/OK/S0001/<category>_0001_OK_C1_<ts>.jpg
    realiad_<res>/<category>/NG/<code>/S0001/<...>_C1_<ts>.jpg (+ mask .png)
    realiad_jsons/realiad_jsons/<category>.json     official train/test split

Access is gated: accept the terms at https://huggingface.co/datasets/Real-IAD/Real-IAD
and set ``HF_TOKEN``. License CC BY-NC-SA 4.0 (research use).
"""

import json
import logging
import re
import shutil
import zlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

from inspection.taxonomy import load_taxonomy
from ml.datasets.common import DEFECT, GOOD, Sample, extract_zip, processed_dir, raw_dir, rel, write_dataset

log = logging.getLogger(__name__)

SOURCE = "realiad"
REPO_ID = "Real-IAD/Real-IAD"
POLYMER_CLASSES = ("plastic_nut", "plastic_plug", "end_cap", "u_block", "mounts")
RESOLUTIONS = ("256", "512", "1024", "raw")
DEFAULT_RESOLUTION = "512"
SPLIT_SET = "realiad_jsons"  # official multi-view setting; see the dataset README for the others

_VIEW_RE = re.compile(r"_C(\d+)_")
_IMG_EXT = {".jpg", ".jpeg", ".png"}


def _fetch(filename: str, downloads: Path) -> Path:
    try:
        return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", local_dir=downloads))
    except (GatedRepoError, HfHubHTTPError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(e, GatedRepoError) or status in (401, 403):
            raise PermissionError(
                "Real-IAD download refused. Check that HF_TOKEN holds a valid Hugging Face read token and "
                f"that access was granted at https://huggingface.co/datasets/{REPO_ID}"
            ) from e
        raise


def download(
    classes=POLYMER_CLASSES, resolution: str = DEFAULT_RESOLUTION, keep_archives: bool = False
) -> None:
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {RESOLUTIONS}")
    raw = raw_dir(SOURCE)
    downloads = raw / "_downloads"

    if not any(raw.glob(f"**/{SPLIT_SET}/*.json")):
        archive = _fetch("realiad_jsons.zip", downloads)
        extract_zip(archive, raw)

    res_dir = raw / f"realiad_{resolution}"
    for cls in classes:
        target = res_dir / cls
        if target.is_dir() and any(target.iterdir()):
            log.info("realiad %s/%s already present", resolution, cls)
            continue
        archive = _fetch(f"realiad_{resolution}/{cls}.zip", downloads)
        tmp = res_dir / f".{cls}.extract"
        shutil.rmtree(tmp, ignore_errors=True)
        extract_zip(archive, tmp)
        # The archive may or may not wrap the category in extra folders.
        root = next(
            (d for d in [tmp, *sorted(tmp.rglob("*"))] if (d / "OK").is_dir() or (d / "NG").is_dir()), None
        )
        if root is None:
            raise FileNotFoundError(f"no OK/ or NG/ folder inside {archive}")
        shutil.rmtree(target, ignore_errors=True)
        root.rename(target)
        shutil.rmtree(tmp, ignore_errors=True)
        if not keep_archives:
            archive.unlink()


def _find_split_json(category: str, split_set: str) -> Path | None:
    hits = sorted(raw_dir(SOURCE).glob(f"**/{split_set}/{category}.json"))
    return hits[0] if hits else None


def _entries_from_json(path: Path) -> list[tuple[str, dict]]:
    meta = json.loads(path.read_text(encoding="utf-8"))
    return [(split, e) for split in ("train", "test") for e in meta.get(split, [])]


def _entries_from_folders(cat_dir: Path) -> list[tuple[str, dict]]:
    """Fallback when the split JSON is missing: OK samples 80/20 train/test, NG all test."""
    entries = []
    for img in sorted(p for p in cat_dir.rglob("*") if p.suffix.lower() in _IMG_EXT):
        parts = img.relative_to(cat_dir).parts
        if img.suffix.lower() == ".png" and parts[0] == "NG":
            continue  # masks
        status = parts[0]
        anomaly = "OK" if status == "OK" else parts[1]
        sample_dir = "/".join(parts[:-1])
        split = "test" if status != "OK" or zlib.crc32(sample_dir.encode()) % 5 == 0 else "train"
        entries.append((split, {"image_path": "/".join(parts), "anomaly_class": anomaly}))
    return entries


def _find_mask(img: Path) -> Path | None:
    for cand in (img.with_suffix(".png"), img.with_name(img.stem + "_mask.png")):
        if cand != img and cand.exists():
            return cand
    return None


def convert(
    classes=POLYMER_CLASSES, resolution: str = DEFAULT_RESOLUTION, split_set: str = SPLIT_SET
) -> list[Path]:
    tax = load_taxonomy()
    outputs = []
    for cls in classes:
        cat_dir = raw_dir(SOURCE) / f"realiad_{resolution}" / cls
        if not cat_dir.is_dir():
            raise FileNotFoundError(f"{cat_dir} not found; run the realiad download first")
        split_json = _find_split_json(cls, split_set)
        if split_json:
            entries = _entries_from_json(split_json)
        else:
            log.warning("no %s/%s.json found; using a deterministic folder-based split", split_set, cls)
            entries = _entries_from_folders(cat_dir)

        samples, missing = [], 0
        for split, e in entries:
            img = _resolve_entry(cat_dir, e["image_path"])
            if img is None:
                missing += 1
                continue
            mapping = tax.map_label(SOURCE, e["anomaly_class"])
            mask = _resolve_entry(cat_dir, e["mask_path"]) if e.get("mask_path") else None
            if mask is None and not mapping.is_good:
                mask = _find_mask(img)
            view = _VIEW_RE.search(img.name)
            samples.append(
                Sample(
                    image_id=f"{SOURCE}/{cls}/{img.stem}",
                    source=SOURCE,
                    category=cls,
                    split=split,
                    image_path=rel(img),
                    label=GOOD if mapping.is_good else DEFECT,
                    zd_class=mapping.zd_class,
                    zd_subtype=mapping.subtype,
                    source_label=e["anomaly_class"],
                    mask_path=rel(mask) if mask else None,
                    sample_id=f"{SOURCE}/{cls}/{img.parent.relative_to(cat_dir).as_posix()}",
                    view=f"C{view.group(1)}" if view else None,
                    extra={"resolution": resolution},
                )
            )
        if missing:
            log.warning("%s: %d images listed in the split file are missing on disk", cls, missing)
        outputs.append(write_dataset(processed_dir(SOURCE, cls), samples))
    return outputs


def _resolve_entry(cat_dir: Path, path: str) -> Path | None:
    """Split-file paths are relative to the category folder; tolerate a leading category name."""
    for cand in (cat_dir / path, cat_dir.parent / path):
        if cand.is_file():
            return cand
    return None
