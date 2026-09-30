"""Train, evaluate and export the YOLO defect detector.

Uses Ultralytics (AGPL-3.0; fine for this academic project). Install it with ``uv sync --group train``.
Runs go to ``outputs/detector/<name>``; evaluation results go next to the weights' run folder.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from inspection.paths import REPO_ROOT

log = logging.getLogger(__name__)

RUNS_DIR = REPO_ROOT / "outputs" / "detector"
DEFAULT_MODEL = "yolo26s.pt"  # NMS-free, fast on CPU; yolo26n.pt for quick runs, yolo11s.pt also works
# On top of the Ultralytics defaults (HSV, translate, scale, horizontal flip, mosaic):
# parts can lie in any orientation on the line.
AUGMENT = {"flipud": 0.5, "degrees": 10.0}
PREDICT_CONF = 0.01  # keep weak detections so line metrics can be computed at any threshold


def _yolo(weights):
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("the detector needs Ultralytics: uv sync --group train") from e
    return YOLO(str(weights))


def train(
    data: Path,
    model: str = DEFAULT_MODEL,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str | None = None,
    name: str = "detector",
    **overrides,
) -> Path:
    """Fine-tune a COCO-pretrained YOLO model on a built dataset; returns the best weights."""
    yolo = _yolo(model)
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RUNS_DIR),
        name=name,
        **{**AUGMENT, "patience": 30, **overrides},
    )
    return Path(yolo.trainer.best)


def _auroc(scores: np.ndarray, positive: np.ndarray) -> float | None:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def line_metrics(pred: pd.DataFrame, conf: float) -> dict:
    """Pass/fail quality at one confidence threshold, per image and per physical part.

    An image fails when any detection reaches ``conf``. A part fails when at least k of its views
    fail (k = 1 .. views per part), which is the multi-view vote of pipeline stage 6; a part with
    fewer than k views needs all of them. False-reject rate: good ones failed. Escape rate:
    defective ones passed.
    """
    defect = pred["n_boxes"].to_numpy() > 0
    flagged = pred["max_conf"].to_numpy() >= conf

    def rates(is_defect, is_flagged) -> dict:
        good = ~is_defect
        return {
            "good": int(good.sum()),
            "defective": int(is_defect.sum()),
            "false_reject_rate": float(is_flagged[good].mean()) if good.any() else None,
            "escape_rate": float((~is_flagged[is_defect]).mean()) if is_defect.any() else None,
        }

    out = {
        "conf": conf,
        "image": {**rates(defect, flagged), "auroc": _auroc(pred["max_conf"].to_numpy(), defect)},
    }
    parts = (
        pred.assign(defect=defect, flagged=flagged)
        .groupby("part")
        .agg(defect=("defect", "any"), flagged_views=("flagged", "sum"), views=("file", "size"))
    )
    max_views = int(parts["views"].max())
    if max_views > 1:
        out["part"] = {
            f"min_views_{k}": rates(
                parts["defect"].to_numpy(),
                parts["flagged_views"].to_numpy() >= np.minimum(k, parts["views"].to_numpy()),
            )
            for k in range(1, max_views + 1)
        }
    return out


def predict_index(weights, data: Path, split: str, imgsz: int, device=None, batch: int = 16) -> pd.DataFrame:
    """Strongest detection per image of one split, joined to the dataset index."""
    yolo = _yolo(weights)
    index = pd.read_csv(data.parent / "index.csv")
    index = index[index["split"] == split].reset_index(drop=True)
    files = [str(data.parent / f) for f in index["file"]]
    max_conf, top_class = np.zeros(len(files)), [None] * len(files)
    for i in range(0, len(files), 256):
        results = yolo.predict(
            files[i : i + 256], conf=PREDICT_CONF, imgsz=imgsz, device=device, batch=batch, verbose=False
        )
        for j, r in enumerate(results, start=i):
            if len(r.boxes):
                k = int(r.boxes.conf.argmax())
                max_conf[j] = float(r.boxes.conf[k])
                top_class[j] = r.names[int(r.boxes.cls[k])]
    return index.assign(max_conf=max_conf, top_class=top_class)


def evaluate(
    weights: Path,
    data: Path,
    split: str = "test",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    batch: int = 16,
) -> dict:
    """mAP and per-class precision / recall / F1, plus line-level pass/fail metrics.

    Writes ``metrics_<split>.json`` and ``predictions_<split>.csv`` into ``<run>/eval/``.
    """
    weights = Path(weights).resolve()  # Ultralytics puts relative output folders under runs/
    out_dir = weights.parent.parent / "eval"
    m = _yolo(weights).val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(out_dir),
        name=f"val_{split}",
        exist_ok=True,
        plots=True,
    )
    pred = predict_index(weights, data, split, imgsz, device, batch)
    result = {
        "weights": str(weights),
        "data": str(data),
        "split": split,
        "imgsz": imgsz,
        "overall": {k: float(v) for k, v in m.results_dict.items()},
        "per_class": [
            {k: v.item() if hasattr(v, "item") else v for k, v in row.items()} for row in m.summary()
        ],
        "speed_ms_per_image": {k: float(v) for k, v in m.speed.items()},
        "line": line_metrics(pred, conf),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_dir / f"predictions_{split}.csv", index=False)
    (out_dir / f"metrics_{split}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("wrote %s", out_dir / f"metrics_{split}.json")
    return result


def export(weights: Path, formats=("onnx",), imgsz: int = 640) -> list[Path]:
    """Export for CPU inference (ONNX Runtime, or OpenVINO when installed).

    ``nms=False`` keeps the NMS-free head of YOLO26, so the graph outputs final detections
    (1, 300, 6: x1, y1, x2, y2, confidence, class) and the app needs no NMS code.
    """
    yolo = _yolo(weights)
    return [Path(yolo.export(format=f, imgsz=imgsz, nms=False)) for f in formats]
