"""Build the no-install web version: the same UI, with the defect model running in the browser.

Exports the trained PatchCore models for TensorFlow.js (ResNet-18 weights with batch norm
folded in, memory banks and defect-type examples, float16), a set of held-out demo test parts,
and the placeholder production data. backend/static/webshim.js answers the UI's /api/* calls
in the browser from these files.

    uv run python -m backend.export_web      -> outputs/showcase.html, outputs/model/*.txt, outputs/data/
"""

import argparse
import base64
import json
import re
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app import PARAMS, Plant, create_app
from inspection.patchcore import MIN_VIEWS, Inspector, calibration_half
from inspection.paths import REPO_ROOT, data_root
from ml.datasets import demo
from ml.datasets.common import load_manifests, resolve

STATIC = Path(__file__).parent / "static"
TFJS = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js"
SHIPPED = {
    "demo": (8, 2),
    "mvtec_ad": (4, 1),
}  # held-out test parts shipped with the page: good, per defect class
# MVTec AD screw: image AUROC 0.87 at 224 px, most defects missed; it needs a higher-resolution model first.
LEFT_OUT = {"screw"}
N_RECORDS, N_LIVE = 3000, 600  # placeholder production records
RECORD_COLS = [
    "part_id",
    "timestamp",
    "part_type",
    "true_class",
    "shift_date",
    "shift",
    "line_id",
    "machine_id",
    "mould_id",
    "cavity",
    "material",
    "operator_id",
    "supplier_id",
    "resin_lot",
    *PARAMS,
]


def _f16(a) -> bytes:
    return np.asarray(a, dtype=np.float16).tobytes()


def _write_b64(path: Path, data: bytes) -> None:
    """Binary data as base64 text: the artifact host only serves text, image and media types."""
    path.write_text(base64.b64encode(data).decode("ascii"), encoding="ascii")


def export_backbone(inspector: Inspector) -> tuple[bytes, dict]:
    """ResNet-18 up to layer3, batch norm folded into each convolution; weights HWIO, then bias."""
    bb = inspector.backbone
    layers = {"conv1": (bb.stem[0], bb.stem[1])}
    for name, layer in (("layer1", bb.stem[4]), ("layer2", bb.layer2), ("layer3", bb.layer3)):
        for i, block in enumerate(layer):
            layers[f"{name}.{i}.conv1"] = (block.conv1, block.bn1)
            layers[f"{name}.{i}.conv2"] = (block.conv2, block.bn2)
            if block.downsample is not None:
                layers[f"{name}.{i}.downsample"] = (block.downsample[0], block.downsample[1])
    parts, spec, off = [], {}, 0
    with torch.no_grad():
        for name, (conv, bn) in layers.items():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = (conv.weight * scale[:, None, None, None]).permute(2, 3, 1, 0).numpy()
            b = (bn.bias - bn.running_mean * scale).numpy()
            spec[name] = {"shape": list(w.shape), "w": off, "b": off + w.size}
            parts += [w.ravel(), b]
            off += w.size + b.size
    return _f16(np.concatenate(parts)), spec


def pick_samples(manifest, part_types) -> list[dict]:
    test = manifest[(manifest.split == "test") & manifest.category.isin(part_types)]
    has_train_defects = set(manifest[(manifest.split == "train") & (manifest.label == "defect")].source)
    test = test[test.source.isin(has_train_defects) | ~calibration_half(test)]  # only never-seen parts
    chosen = []
    for (source, pt), df in test.groupby(["source", "category"]):
        n_good, n_defect = SHIPPED[source]
        per_sample = df.groupby("sample_id").zd_class.first()
        ids = list(per_sample[per_sample.isna()].index[:n_good])
        for _, g in per_sample.dropna().groupby(per_sample.dropna()):
            ids += list(g.index[:n_defect])
        for sid in ids:
            views = df[df.sample_id == sid].sort_values("view")
            label = views.zd_class.dropna()
            chosen.append(
                {
                    "id": sid,
                    "part_type": pt,
                    "label": label.iloc[0] if len(label) else None,
                    "paths": list(views.image_path),
                    "views": [
                        v if isinstance(v, str) else None for v in views.view
                    ],  # MVTec: one unnamed view
                }
            )
    return chosen


def build_page(title: str, tfjs_src: str) -> str:
    """The app page as an artifact fragment (no html/head/body tags), with the in-browser backend."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    head = re.search(r"<head>(.*)</head>", html, re.S)[1]
    body = re.search(r"<body>(.*)</body>", html, re.S)[1]
    head = re.sub(r"<meta [^>]*>\s*", "", head).replace(
        "<title>ZeroDefect</title>", f"<title>{title}</title>"
    )
    shim = (STATIC / "webshim.js").read_text(encoding="utf-8")
    body = body.replace(
        "<script>", f'<script src="{tfjs_src}"></script>\n<script>\n{shim}\n</script>\n<script>', 1
    )
    return head.strip() + "\n" + body.strip() + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m backend.export_web",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--title", default="ZeroDefect Line Monitor")
    args = p.parse_args(argv)
    out = args.out

    demo_manifest = load_manifests(["demo"])
    if not ((demo_manifest.split == "train") & (demo_manifest.label == "defect")).any():
        demo.generate()
    sources = ["demo"] + (["mvtec_ad"] if not load_manifests(["mvtec_ad"]).empty else [])
    manifest = load_manifests(sources)
    inspector = Inspector.load_or_fit(sources, data_root() / "models")
    inspector.models = {k: m for k, m in inspector.models.items() if k not in LEFT_OUT}
    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    weights, spec = export_backbone(inspector)
    _write_b64(out / "model/backbone.txt", weights)
    parts = {}
    for pt, m in inspector.models.items():
        _write_b64(out / f"model/{pt}.txt", _f16(m.bank.numpy()) + _f16(m.class_vecs.numpy()))
        parts[pt] = {"threshold": m.threshold, "bank": len(m.bank), "labels": m.class_labels}

    samples = pick_samples(manifest, inspector.models)
    blob = bytearray()  # JPEG bytes of every view, back to back (PNG sources re-encoded as JPEG)
    for s in samples:
        views = []
        for view, path in zip(s["views"], s.pop("paths"), strict=True):
            data = resolve(path).read_bytes()
            if path.endswith(".png"):
                data = cv2.imencode(".jpg", cv2.imread(str(resolve(path))), [cv2.IMWRITE_JPEG_QUALITY, 90])[
                    1
                ].tobytes()
            views.append({"view": view, "off": len(blob), "len": len(data)})
            blob += data
        s["views"] = views
    _write_b64(out / "data/images.txt", bytes(blob))

    # Placeholder production data, straight from the real backend's API.
    plant = Plant(("demo",), inspector=inspector, start_line=False)
    client = TestClient(create_app(plant))
    by_class: dict[tuple[str, str | None], list[int]] = {}
    for i, s in enumerate(samples):
        by_class.setdefault((s["part_type"], s["label"]), []).append(i)

    def rows(df):
        out_rows = []
        for rec in df[RECORD_COLS].itertuples(index=False):
            r = [None if isinstance(v, float) and np.isnan(v) else v for v in rec]
            r[1] = r[1].isoformat(timespec="seconds")
            r[4] = str(r[4])
            r[9] = int(r[9])
            r[14:] = [round(float(v), 2) for v in r[14:]]
            pool = by_class.get((r[2], r[3])) or by_class[(r[2], None)]
            out_rows.append([*r, pool[zlib.crc32(r[0].encode()) % len(pool)]])
        return out_rows

    live = plant.sim.run_day().head(N_LIVE)
    data = {
        "config": client.get("/api/config").json(),
        "model": client.get("/api/model").json(),
        "summary": client.get("/api/summary").json(),
        "machines": client.get("/api/machines").json(),
        "history": {m.id: client.get(f"/api/machines/{m.id}/history").json() for _, m in plant.cfg.machines},
        "min_views": MIN_VIEWS,
        "backbone": spec,
        "parts": parts,
        "samples": samples,
        "cols": [*RECORD_COLS, "sample"],
        "records": rows(plant.frame().tail(N_RECORDS)),
        "live": rows(live.assign(true_class=None)),
    }
    (out / "data/demo.json").write_text(
        json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8"
    )

    # Reference results from the Python model, to check the browser gives the same answers.
    expected = []
    for s in samples:
        res = inspector.inspect_part(
            s["part_type"],
            [
                cv2.imdecode(
                    np.frombuffer(bytes(blob[v["off"] : v["off"] + v["len"]]), np.uint8), cv2.IMREAD_COLOR
                )
                for v in s["views"]
            ],
        )
        expected.append(
            {
                "id": s["id"],
                "defect": res["defect"],
                "cls": res["cls"],
                "score": res["score"],
                "view_scores": [v["score"] for v in res["views"]],
            }
        )
    (out / "web_expected.json").write_text(json.dumps(expected), encoding="utf-8")

    (out / "showcase.html").write_text(build_page(args.title, TFJS), encoding="utf-8")
    sizes = {
        p.relative_to(out).as_posix(): round(p.stat().st_size / 1e6, 1)
        for p in [
            out / "showcase.html",
            *sorted((out / "model").iterdir()),
            *sorted((out / "data").iterdir()),
        ]
    }
    print(f"{len(samples)} demo parts, {len(data['records'])} placeholder records. Files (MB): {sizes}")


if __name__ == "__main__":
    main()
