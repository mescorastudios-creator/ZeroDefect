"""MVTec AD 2, class ``wallplugs`` (a polymer part): robustness to lighting.

Layout (``data/raw/mvtec_ad2/<class>``): ``train/good``, ``validation/good``,
``test_public/{good,bad}``, ``test_public/ground_truth/bad/<stem>_mask.png``.
The changing-lighting sets ``test_private`` and ``test_private_mixed`` ship
without ground truth (scored on https://benchmark.mvtec.com/), so they are not
converted. License CC BY-NC-SA 4.0.
"""

from pathlib import Path

from inspection.taxonomy import load_taxonomy
from ml.datasets.common import (
    DEFECT,
    GOOD,
    Sample,
    extract_classes_from_tar,
    processed_dir,
    raw_dir,
    rel,
    write_dataset,
)

SOURCE = "mvtec_ad2"
CLASSES = ("wallplugs",)
# Full archive (32.7 GB), the link anomalib uses. The per-class archive link from
# the MVTec download form is much smaller: pass it with --url.
ARCHIVE_URL = (
    "https://www.mydrive.ch/shares/150997/701c90d3aea6588f404936e32a674602/"
    "download/466712769-1743429042/mvtec_ad_2.tar.gz"
)
_FOLDERS = {  # folder -> (split, label)
    "train/good": ("train", "good"),
    "validation/good": ("val", "good"),
    "test_public/good": ("test", "good"),
    "test_public/bad": ("test", "bad"),
}


def download(classes=CLASSES, url: str = ARCHIVE_URL) -> None:
    missing = [c for c in classes if not (raw_dir(SOURCE) / c).is_dir()]
    if missing:
        extract_classes_from_tar(url, raw_dir(SOURCE), missing)


def convert(classes=CLASSES) -> list[Path]:
    tax = load_taxonomy()
    outputs = []
    for cls in classes:
        root = raw_dir(SOURCE) / cls
        if not root.is_dir():
            raise FileNotFoundError(f"{root} not found; run the mvtec_ad2 download first")
        samples = []
        for folder, (split, label) in _FOLDERS.items():
            mapping = tax.map_label(SOURCE, label)
            for img in sorted((root / folder).glob("*.png")):
                mask = root / "test_public" / "ground_truth" / "bad" / f"{img.stem}_mask.png"
                image_id = f"{SOURCE}/{cls}/{folder}/{img.stem}"
                samples.append(
                    Sample(
                        image_id=image_id,
                        source=SOURCE,
                        category=cls,
                        split=split,
                        image_path=rel(img),
                        label=GOOD if mapping.is_good else DEFECT,
                        zd_class=mapping.zd_class,
                        source_label=label,
                        mask_path=rel(mask) if label == "bad" and mask.exists() else None,
                        sample_id=image_id,
                    )
                )
        outputs.append(write_dataset(processed_dir(SOURCE, cls), samples))
    return outputs
