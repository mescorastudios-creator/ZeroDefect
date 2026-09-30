"""ZeroDefect web app: live AI defect inspection, image upload, dashboard, factory map and traceability.

Defect finding is real: every part on the live line (and every uploaded image) goes through the
PatchCore model. The production context around it (machines, operators, resin lots, process
values, the history behind the dashboard) is PLACEHOLDER data from the plant simulator.
"""

import asyncio
import json
import random
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from inspection.patchcore import MIN_VIEWS, Inspector, heatmap_png
from inspection.paths import data_root
from inspection.taxonomy import load_taxonomy
from ml.datasets.common import resolve
from simulator.config import load_config
from simulator.images import GOOD, ImagePool
from simulator.production import ProductionSimulator

STATIC = Path(__file__).parent / "static"
PARAMS = ["melt_temp", "injection_pressure", "holding_time", "cooling_time", "mould_temp", "cycle_time"]
WINDOW = timedelta(hours=1)  # "recent" window for machine status and alerts, in simulated time
GOOD_SHARE = 0.6  # share of good parts the live camera feeds in; the rest are defective test parts
MAX_UPLOAD = 20 * 1024 * 1024


def _clean(v):
    """JSON-safe scalar: NaN -> None, numpy -> Python, timestamps -> ISO."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat(timespec="seconds")
    return v.item() if isinstance(v, np.generic) else v


def _result(res: dict, ms: float) -> dict:
    keys = ("defect", "cls", "confidence", "score", "threshold", "flagged_views")
    return {**{k: res[k] for k in keys if k in res}, "ms": round(ms)}


class Plant:
    """Placeholder production history up to now, then a live line where the model inspects every part."""

    def __init__(self, images=("demo",), history_days: int = 7, interval: float = 1.5, inspector=None):
        self.cfg = load_config()
        self.pool = ImagePool.from_sources(images)
        self.samples = sorted(self.pool.views)
        self.sample_index = {s: i for i, s in enumerate(self.samples)}
        self.inspector = inspector or Inspector.load_or_fit(images[0], data_root() / "models")
        self.interval = interval
        start = (datetime.now() - timedelta(days=history_days)).replace(second=0, microsecond=0)
        self.sim = ProductionSimulator(self.cfg, self.pool, start=start)
        self.df = self.sim.run(history_days)
        daily = (self.df.true_label == "defect").groupby([self.df.machine_id, self.df.shift_date]).mean()
        self.baseline = daily.groupby(level=0).median().to_dict()  # a machine's normal defect rate
        self.events: deque[tuple[int, dict]] = deque(maxlen=200)
        self.seq = 0
        self._inspected: dict[str, dict] = {}  # recent inspections by part id, with heat maps
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._line, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def _line(self) -> None:
        """The live camera: next part, inspect all its views, publish, repeat."""
        rng = np.random.default_rng()
        for rec in self.sim.iter_parts():
            if self._stop.is_set():
                return
            t0 = time.monotonic()
            pt = rec["part_type"]
            defects = sorted(self.pool.classes(pt) - {GOOD})
            cls = GOOD if rng.random() < GOOD_SHARE or not defects else str(rng.choice(defects))
            rec.update(
                image_sample_id=str(self.pool.sample(pt, cls, 1, rng)[0]),
                true_label="good" if cls == GOOD else "defect",
                true_class=None if cls == GOOD else cls,
            )
            event = self.inspect(rec)
            with self._lock:
                self._pending.append(rec)
                self.seq += 1
                self.events.append((self.seq, event))
            time.sleep(max(0.0, self.interval - (time.monotonic() - t0)))

    def inspect(self, rec: dict) -> dict:
        """Run the model on all camera views of a part (cached for recent parts)."""
        if hit := self._inspected.get(rec["part_id"]):
            return hit
        views = self.pool.views[rec["image_sample_id"]]
        t = time.perf_counter()
        res = self.inspector.inspect_part(
            rec["part_type"], [cv2.imread(str(resolve(v["image_path"]))) for v in views]
        )
        out = self.part(rec)
        out["result"] = _result(res, 1000 * (time.perf_counter() - t))
        for v, r in zip(out["views"], res["views"], strict=True):
            v.update(
                score=r["score"],
                defect=r["defect"],
                cls=r["cls"],
                confidence=r["confidence"],
                boxes=r["boxes"],
                heatmap=heatmap_png(r["heatmap"], res["threshold"]),
            )
        with self._lock:
            self._inspected[rec["part_id"]] = out
            while len(self._inspected) > 300:
                self._inspected.pop(next(iter(self._inspected)))
        return out

    def frame(self) -> pd.DataFrame:
        """All parts so far (placeholder history + live), in time order."""
        with self._lock:
            if self._pending:
                self.df = pd.concat([self.df, pd.DataFrame(self._pending)], ignore_index=True)
                self._pending = []
            return self.df

    def part(self, rec) -> dict:
        sid = rec["image_sample_id"]
        k = self.sample_index[sid]
        return {
            "id": rec["part_id"],
            "ts": _clean(rec["timestamp"]),
            "part_type": rec["part_type"],
            "label": _clean(rec["true_class"]),  # dataset label of the image, to check the model against
            "context": {  # placeholder production data
                "shift_date": str(rec["shift_date"]),
                "shift": rec["shift"],
                "line": rec["line_id"],
                "machine": rec["machine_id"],
                "mould": rec["mould_id"],
                "cavity": _clean(rec["cavity"]),
                "material": rec["material"],
                "operator": rec["operator_id"],
                "supplier": rec["supplier_id"],
                "lot": rec["resin_lot"],
                "params": {p: _clean(rec[p]) for p in PARAMS},
            },
            "views": [
                {"view": v["view"], "url": f"/api/samples/{k}/{i}"}
                for i, v in enumerate(self.pool.views[sid])
            ],
        }

    def machines(self) -> list[dict]:
        """Placeholder machine status over the last simulated hour, with rule-based alerts."""
        df = self.frame()
        recent = df.iloc[
            np.searchsorted(df.timestamp.to_numpy(), np.datetime64(df.timestamp.iloc[-1] - WINDOW)) :
        ]
        classes = {c.key: c.name for c in load_taxonomy().classes}
        out = []
        for line, m in self.cfg.machines:
            r = recent[recent.machine_id == m.id]
            n, bad = len(r), int((r.true_label == "defect").sum())
            rate, base = (bad / n if n else 0.0), self.baseline.get(m.id, 0.0)
            alerts = []
            if n and bad >= 3 and rate >= 2 * base:
                top = r.true_class.dropna().value_counts().index[0]
                alerts.append(
                    f"Defect rate {rate:.1%} is {rate / base:.1f}× normal, mostly {classes[top].lower()}"
                )
            means = {p: float(r[p].mean()) if n else None for p in PARAMS}
            for p, spec in self.cfg.process_params.items():
                if n and abs(means[p] - m.process[p]) > 0.6 * spec.tolerance:
                    name = p.replace("_", " ").capitalize()
                    alerts.append(
                        f"{name} {means[p]:.1f} {spec.unit}, {means[p] - m.process[p]:+.1f} "
                        f"from set point {m.process[p]:g}"
                    )
            last = df[df.machine_id == m.id].iloc[-1]
            status = "stopped" if n == 0 else "alarm" if alerts else "watch" if rate > 1.5 * base else "ok"
            out.append(
                {
                    "id": m.id,
                    "line": line.id,
                    "line_name": line.name,
                    "part_type": m.part_type,
                    "material": m.material,
                    "status": status,
                    "parts_per_hour": n,
                    "defects": bad,
                    "rate": rate,
                    "baseline": base,
                    "operator": last.operator_id,
                    "lot": last.resin_lot,
                    "means": means,
                    "setpoints": m.process,
                    "alerts": alerts,
                }
            )
        return out


def create_app(plant: Plant) -> FastAPI:
    app = FastAPI(title="ZeroDefect")
    tax = load_taxonomy()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/config")
    def config():
        return {
            "classes": {
                c.key: {"name": c.name, "color": c.color, "severity": c.default_severity} for c in tax.classes
            },
            "params": {
                p: {"unit": s.unit, "tolerance": s.tolerance} for p, s in plant.cfg.process_params.items()
            },
            "machines": {
                m.id: {"line": line.id, "part_type": m.part_type, "setpoints": m.process}
                for line, m in plant.cfg.machines
            },
            "part_types": sorted(plant.inspector.models),
        }

    @app.get("/api/model")
    def model():
        return {
            "method": "PatchCore anomaly detection + k-nearest-neighbour defect typing",
            "backbone": "ResNet-18 (ImageNet), layer 2+3 features, 224 px",
            "min_views": MIN_VIEWS,
            "part_types": {
                k: {"threshold": m.threshold, **m.metrics} for k, m in plant.inspector.models.items()
            },
        }

    @app.get("/api/stream")
    async def stream():
        async def events():
            last = plant.seq - 6  # a few recent parts so the screen is never empty
            while True:
                for seq, event in list(plant.events):
                    if seq > last:
                        last = seq
                        yield f"data: {json.dumps(event)}\n\n"
                await asyncio.sleep(0.2)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/api/inspect")
    async def inspect_upload(request: Request, part_type: str = "auto"):
        body = await request.body()
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "Image is larger than 20 MB")
        img = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "Not an image file (use JPEG, PNG or BMP)")
        if part_type != "auto" and part_type not in plant.inspector.models:
            raise HTTPException(400, f"Unknown part type {part_type!r}")
        scale = 1024 / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        t = time.perf_counter()
        res = await asyncio.to_thread(
            plant.inspector.inspect_image, img, None if part_type == "auto" else part_type
        )
        view = res["views"][0]
        return {
            "part_type": res["part_type"],
            **_result(res, 1000 * (time.perf_counter() - t)),
            "boxes": view["boxes"],
            "heatmap": heatmap_png(view["heatmap"], res["threshold"]),
        }

    @app.get("/api/test-image")
    def test_image(defect: bool = False):
        """A random held-out test view, with its dataset label in the X-Label header."""
        cands = [s for s, c in plant.pool.sample_class.items() if (c != GOOD) == defect]
        sid = random.choice(cands)
        view = random.choice(plant.pool.views[sid])
        return FileResponse(
            resolve(view["image_path"]),
            headers={
                "X-Label": plant.pool.sample_class[sid],
                "X-Part-Type": sid.split("/")[1],
                "Cache-Control": "no-store",
            },
        )

    @app.get("/api/summary")
    def summary():
        df = plant.frame()
        bad = df.true_label == "defect"
        return {
            "parts": len(df),
            "defects": int(bad.sum()),
            "days": df.shift_date.nunique(),
            "pareto": [{"k": k, "n": int(n)} for k, n in df.true_class[bad].value_counts().items()],
            "machines": [{"k": k, "v": float(v)} for k, v in bad.groupby(df.machine_id).mean().items()],
            "yield": [
                {"d": str(d), "v": 100 * (1 - float(v))} for d, v in bad.groupby(df.shift_date).mean().items()
            ],
        }

    @app.get("/api/machines")
    def machines():
        return {"machines": plant.machines()}

    @app.get("/api/machines/{machine_id}/history")
    def machine_history(machine_id: str):
        df = plant.frame()
        m = df[df.machine_id == machine_id]
        if m.empty:
            raise HTTPException(404, "unknown machine")
        g = m.groupby("shift_date")
        rate = (m.true_label == "defect").groupby(m.shift_date).mean()
        return {
            "days": [str(d) for d in rate.index],
            "rate": [100 * float(v) for v in rate],
            "params": {p: [float(v) for v in g[p].mean()] for p in PARAMS},
        }

    @app.get("/api/search")
    def search(q: str = "", machine: str = "", result: str = "", limit: int = 100):
        df = plant.frame()
        mask = np.ones(len(df), bool)
        if q.strip():
            q = q.strip().upper()
            mask &= (
                df.part_id.str.contains(q, regex=False)
                | df.resin_lot.str.contains(q, regex=False)
                | (df.operator_id == q)
            ).to_numpy()
        if machine:
            mask &= (df.machine_id == machine).to_numpy()
        if result in ("good", "defect"):
            mask &= (df.true_label == result).to_numpy()
        hits = df[mask].iloc[::-1].head(min(limit, 500))
        cols = ["part_id", "timestamp", "machine_id", "cavity", "operator_id", "resin_lot", "true_class"]
        return {
            "total": int(mask.sum()),
            "rows": [
                {c: _clean(v) for c, v in zip(cols, row, strict=True)}
                for row in hits[cols].itertuples(index=False)
            ],
        }

    @app.get("/api/parts/{part_id}")
    def part(part_id: str):
        df = plant.frame()
        hit = df[df.part_id == part_id]
        if hit.empty:
            raise HTTPException(404, "unknown part")
        return plant.inspect(hit.iloc[-1].to_dict())  # runs the model if the part was not inspected yet

    @app.get("/api/samples/{k}/{i}")
    def image(k: int, i: int):
        if not 0 <= k < len(plant.samples):
            raise HTTPException(404, "unknown sample")
        views = plant.pool.views[plant.samples[k]]
        if not 0 <= i < len(views):
            raise HTTPException(404, "unknown view")
        return FileResponse(resolve(views[i]["image_path"]), headers={"Cache-Control": "max-age=86400"})

    return app
