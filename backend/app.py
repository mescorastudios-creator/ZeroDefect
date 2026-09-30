"""ZeroDefect web app: a live simulated line with inspection, dashboard, factory map and traceability.

The plant simulator produces parts in (accelerated) real time; the UI streams them over
Server-Sent Events. Verdicts are the dataset labels until the M1 inspection model exists.
"""

import asyncio
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from inspection.paths import data_root
from inspection.taxonomy import load_taxonomy
from ml.datasets.common import resolve
from simulator.config import load_config
from simulator.images import ImagePool
from simulator.production import ProductionSimulator

STATIC = Path(__file__).parent / "static"
PARAMS = ["melt_temp", "injection_pressure", "holding_time", "cooling_time", "mould_temp", "cycle_time"]
WINDOW = timedelta(hours=1)  # "recent" window for machine status and alerts, in simulated time


def _clean(v):
    """JSON-safe scalar: NaN -> None, numpy -> Python, timestamps -> ISO."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat(timespec="seconds")
    return v.item() if isinstance(v, np.generic) else v


def coco_boxes(sources) -> dict[str, list[dict]]:
    """Defect boxes per image path, in % of the image size, from the converted COCO files."""
    tax = load_taxonomy()
    out: dict[str, list[dict]] = {}
    for path in (p for s in sources for p in (data_root() / "processed" / s).glob("*/coco.json")):
        coco = json.loads(path.read_text(encoding="utf-8"))
        images = {im["id"]: im for im in coco["images"]}
        for a in coco["annotations"]:
            im = images[a["image_id"]]
            x, y, w, h = a["bbox"]
            out.setdefault(im["file_name"], []).append(
                {
                    "x": 100 * x / im["width"],
                    "y": 100 * y / im["height"],
                    "w": 100 * w / im["width"],
                    "h": 100 * h / im["height"],
                    "cls": tax.by_id(a["category_id"]).key,
                }
            )
    return out


class Plant:
    """The running line: simulated history up to now, then live production in a background thread."""

    def __init__(self, images=("demo",), history_days: int = 7, speed: float = 5.0):
        self.cfg = load_config()
        self.pool = ImagePool.from_sources(images)
        self.samples = sorted(self.pool.views)
        self.sample_index = {s: i for i, s in enumerate(self.samples)}
        self.boxes = coco_boxes(images)
        self.speed = speed
        start = (datetime.now() - timedelta(days=history_days)).replace(second=0, microsecond=0)
        self.sim = ProductionSimulator(self.cfg, self.pool, start=start)
        self.df = self.sim.run(history_days)
        daily = (self.df.true_label == "defect").groupby([self.df.machine_id, self.df.shift_date]).mean()
        self.baseline = daily.groupby(level=0).median().to_dict()  # a machine's normal defect rate
        self.live: list[dict] = []
        self._pending: list[dict] = []
        self._lock = threading.Lock()
        self._t0_sim, self._t0_wall = start + timedelta(days=history_days), time.monotonic()
        threading.Thread(target=self._produce, daemon=True).start()

    def now(self) -> datetime:
        return self._t0_sim + timedelta(seconds=(time.monotonic() - self._t0_wall) * self.speed)

    def _produce(self) -> None:
        for rec in self.sim.iter_parts():
            wait = (rec["timestamp"] - self.now()).total_seconds() / self.speed
            if wait > 0:
                time.sleep(wait)
            with self._lock:
                self.live.append(rec)
                self._pending.append(rec)

    def frame(self) -> pd.DataFrame:
        """All parts so far (history + live), in time order."""
        with self._lock:
            if self._pending:
                self.df = pd.concat([self.df, pd.DataFrame(self._pending)], ignore_index=True)
                self._pending = []
            return self.df

    def recent(self, df: pd.DataFrame) -> pd.DataFrame:
        i = np.searchsorted(df.timestamp.to_numpy(), np.datetime64(self.now() - WINDOW))
        return df.iloc[i:]

    def part(self, rec) -> dict:
        sid = rec["image_sample_id"]
        k = self.sample_index[sid]
        return {
            "id": rec["part_id"],
            "ts": _clean(rec["timestamp"]),
            "shift_date": str(rec["shift_date"]),
            "shift": rec["shift"],
            "line": rec["line_id"],
            "machine": rec["machine_id"],
            "mould": rec["mould_id"],
            "cavity": _clean(rec["cavity"]),
            "part_type": rec["part_type"],
            "material": rec["material"],
            "operator": rec["operator_id"],
            "supplier": rec["supplier_id"],
            "lot": rec["resin_lot"],
            "params": {p: _clean(rec[p]) for p in PARAMS},
            "cls": _clean(rec["true_class"]),
            "views": [
                {
                    "view": v["view"],
                    "url": f"/api/samples/{k}/{i}",
                    "boxes": self.boxes.get(v["image_path"], []),
                }
                for i, v in enumerate(self.pool.views[sid])
            ],
        }

    def machines(self) -> list[dict]:
        """Status per machine over the last simulated hour, with rule-based alerts."""
        df = self.frame()
        recent = self.recent(df)
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
                dev = (means[p] or m.process[p]) - m.process[p]
                if n and abs(dev) > 0.6 * spec.tolerance:
                    name = p.replace("_", " ").capitalize()
                    alerts.append(
                        f"{name} {means[p]:.1f} {spec.unit}, {dev:+.1f} from set point {m.process[p]:g}"
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
            "speed": plant.speed,
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
        }

    @app.get("/api/stream")
    async def stream():
        async def events():
            i = max(len(plant.live) - 12, 0)  # a few recent parts so the screen is never empty
            while True:
                new = plant.live[i:]
                i += len(new)
                for rec in new:
                    yield f"data: {json.dumps(plant.part(rec))}\n\n"
                await asyncio.sleep(0.25)

        return StreamingResponse(
            events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.get("/api/summary")
    def summary():
        df = plant.frame()
        bad = df.true_label == "defect"
        return {
            "now": _clean(plant.now()),
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
        return {"now": _clean(plant.now()), "machines": plant.machines()}

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
        return plant.part(hit.iloc[-1].to_dict())

    @app.get("/api/samples/{k}/{i}")
    def image(k: int, i: int):
        if not 0 <= k < len(plant.samples):
            raise HTTPException(404, "unknown sample")
        views = plant.pool.views[plant.samples[k]]
        if not 0 <= i < len(views):
            raise HTTPException(404, "unknown view")
        return FileResponse(resolve(views[i]["image_path"]), headers={"Cache-Control": "max-age=86400"})

    return app
