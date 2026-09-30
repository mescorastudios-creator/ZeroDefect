"""MVTec AD, classes ``screw`` and ``metal_nut``: the metal nut/bolt extension.

Layout (``data/raw/mvtec_ad/<class>``): ``train/good``, ``test/<defect>``,
``ground_truth/<defect>/<stem>_mask.png``. License CC BY-NC-SA 4.0.
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

SOURCE = "mvtec_ad"
CLASSES = ("screw", "metal_nut")
# Official archive (5.3 GB); only the requested class folders are extracted.
ARCHIVE_URL = (
    "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
    "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
)


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
            raise FileNotFoundError(f"{root} not found; run the mvtec_ad download first")
        samples = []
        for split in ("train", "test"):
            for defect_dir in sorted(d for d in (root / split).iterdir() if d.is_dir()):
                mapping = tax.map_label(SOURCE, defect_dir.name)
                for img in sorted(defect_dir.glob("*.png")):
                    mask = root / "ground_truth" / defect_dir.name / f"{img.stem}_mask.png"
                    image_id = f"{SOURCE}/{cls}/{split}/{defect_dir.name}/{img.stem}"
                    samples.append(
                        Sample(
                            image_id=image_id,
                            source=SOURCE,
                            category=cls,
                            split=split,
                            image_path=rel(img),
                            label=GOOD if mapping.is_good else DEFECT,
                            zd_class=mapping.zd_class,
                            zd_subtype=mapping.subtype,
                            source_label=defect_dir.name,
                            mask_path=rel(mask) if mask.exists() else None,
                            sample_id=image_id,
                        )
                    )
        outputs.append(write_dataset(processed_dir(SOURCE, cls), samples))
    return outputs
