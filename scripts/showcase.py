"""Build the presentation page: a replay of simulated production on real MVTec AD photos.

Defect boxes come from the dataset annotations; the AI inspection model plugs in at M1.
Needs MVTec AD: `uv run python -m ml.datasets download mvtec_ad --convert`.

    uv run python scripts/showcase.py            -> outputs/showcase.html
"""

import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from inspection.paths import REPO_ROOT, data_root
from inspection.taxonomy import load_taxonomy
from ml.datasets.common import resolve
from simulator.config import load_config
from simulator.images import ImagePool
from simulator.production import ProductionSimulator

# MVTec AD has screws and nuts (the nut-bolt line from the abstract); IMM-03 makes screws so its drift shows.
MACHINE_PARTS = {
    "IMM-01": "metal_nut",
    "IMM-02": "screw",
    "IMM-03": "screw",
    "IMM-04": "metal_nut",
    "IMM-05": "metal_nut",
    "IMM-06": "screw",
}
PART_NAMES = {"screw": "Screw", "metal_nut": "Metal nut"}
REPLAY_DAY, N_DEFECT, N_GOOD, THUMB = "2026-09-08", 16, 24, 420


def _jpeg(img) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def main(out: Path) -> None:
    cfg = load_config()
    for _, m in cfg.machines:
        m.part_type = MACHINE_PARTS[m.id]
    pool = ImagePool.from_sources(["mvtec_ad"])
    parts = ProductionSimulator(cfg, pool).run()
    tax = load_taxonomy()
    is_defect = parts.true_label == "defect"

    boxes = {}
    for cat in PART_NAMES:
        coco = json.loads((data_root() / f"processed/mvtec_ad/{cat}/coco.json").read_text())
        by_id = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            im = by_id[a["image_id"]]
            x, y, w, h = a["bbox"]
            boxes.setdefault(im["file_name"], []).append(
                {
                    "x": 100 * x / im["width"],
                    "y": 100 * y / im["height"],
                    "w": 100 * w / im["width"],
                    "h": 100 * h / im["height"],
                    "cls": tax.by_id(a["category_id"]).key,
                }
            )

    day = parts[parts.shift_date.astype(str) == REPLAY_DAY]
    rng = np.random.default_rng(1)
    pick = np.concatenate(
        [
            rng.choice(np.flatnonzero(day.true_label == lbl), n, replace=False)
            for lbl, n in (("defect", N_DEFECT), ("good", N_GOOD))
        ]
    )
    live = []
    for _, p in day.iloc[np.sort(pick)].iterrows():
        view = pool.views[p.image_sample_id][0]
        img = cv2.imread(str(resolve(view["image_path"])))
        live.append(
            {
                "img": _jpeg(cv2.resize(img, (THUMB, THUMB), interpolation=cv2.INTER_AREA)),
                "boxes": boxes.get(view["image_path"], []),
                "id": p.part_id,
                "time": f"{p.timestamp:%H:%M:%S}",
                "date": str(p.shift_date),
                "machine": p.machine_id,
                "cavity": int(p.cavity),
                "part": PART_NAMES[p.part_type],
                "operator": p.operator_id,
                "shift": p["shift"],
                "lot": p.resin_lot,
                "supplier": p.supplier_id,
                "params": {
                    k: float(p[k]) for k in ("melt_temp", "injection_pressure", "mould_temp", "cycle_time")
                },
                "cls": None if p.true_label == "good" else p.true_class,
            }
        )

    imm03 = parts[parts.machine_id == "IMM-03"].groupby("shift_date")
    data = {
        "kpi": {
            "parts": len(parts),
            "defects": int(is_defect.sum()),
            "days": cfg.days,
            "machines": len(cfg.machines),
            "lines": len(cfg.lines),
        },
        "classes": {
            c.key: {"name": c.name, "color": c.color, "severity": c.default_severity} for c in tax.classes
        },
        "setpoints": {m.id: m.process for _, m in cfg.machines},
        "pareto": [{"k": k, "n": int(n)} for k, n in parts.true_class[is_defect].value_counts().items()],
        "machines": [
            {"k": k, "part": PART_NAMES[MACHINE_PARTS[k]], "v": 100 * float(v)}
            for k, v in is_defect.groupby(parts.machine_id).mean().items()
        ],
        "yield": [
            {"d": str(d), "v": 100 * (1 - float(v))}
            for d, v in is_defect.groupby(parts.shift_date).mean().items()
        ],
        "imm03_temp": [{"d": str(d), "v": float(v)} for d, v in imm03.mould_temp.mean().items()],
        "imm03_crack": [{"d": str(d), "v": 100 * float((g.true_class == "crack").mean())} for d, g in imm03],
        "live": live,
    }
    html = (REPO_ROOT / "scripts/showcase.html").read_text(encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "outputs/showcase.html")
