"""Production-line simulator.

Generates one traceability record per moulded part: machine, mould cavity,
operator, shift, resin lot and supplier, process parameters, and the true
defect class, optionally bound to a real dataset sample of that class.

Generation runs one machine-day at a time with a random generator seeded by
(seed, machine, day) and state carried between days, so a batch history and a
paced live stream produce identical records.
"""

import logging
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from inspection.taxonomy import load_taxonomy
from simulator.config import Fault, Machine, PlantConfig, Supplier
from simulator.images import GOOD, ImagePool

log = logging.getLogger(__name__)

DAY = 86400.0
MAX_DEFECT_PROB = 0.9


def fault_scope(fault: Fault, t: np.ndarray, ctx: Mapping[str, np.ndarray | str | int]) -> np.ndarray:
    """Boolean mask of the parts a fault applies to.

    ``t`` is seconds since the simulation start; ``ctx`` holds the part columns
    machine_id, cavity, operator_id, supplier_id, resin_lot and shift.
    """
    mask = (t >= fault.from_day * DAY) & (t < fault.to_day * DAY)
    for attr, col in [
        ("machine", "machine_id"),
        ("cavity", "cavity"),
        ("operator", "operator_id"),
        ("supplier", "supplier_id"),
        ("lot", "resin_lot"),
        ("shift", "shift"),
    ]:
        want = getattr(fault, attr)
        if want is not None:
            mask = mask & (np.asarray(ctx[col]) == want)
    return mask


def fault_targets(fault: Fault, cfg: PlantConfig) -> list[str]:
    """Defect classes a fault makes more likely."""
    if fault.kind == "multiplier":
        return [c for c, mult in fault.classes.items() if mult > 1]
    return [
        c
        for c, coefs in cfg.defect_model.process.items()
        if coefs.get(fault.parameter, 0) * fault.per_day > 0
    ]


@dataclass
class _MachineState:
    next_shot: float  # seconds since start
    ar: dict[str, float]
    shot_no: int = 0


@dataclass
class _Lot:
    lot_id: str
    supplier_id: str
    received: datetime
    moisture_pct: float  # incoming-inspection measurement
    moisture: float = field(repr=False)  # hidden quality factors (1 = nominal)
    contamination: float = field(repr=False)


class ProductionSimulator:
    def __init__(self, cfg: PlantConfig, pool: ImagePool | None = None, start: datetime | None = None):
        self.cfg, self.pool = cfg, pool
        self.start = start or cfg.start
        self.day = 0
        self.classes = list(load_taxonomy().keys)
        self.params = list(cfg.process_params)
        self.stoppages: list[dict] = []
        self._lots: dict[tuple[str, int], _Lot] = {}
        self._line_index = {line.id: i for i, line in enumerate(cfg.lines)}
        self._supplier_index = {s.id: i for i, s in enumerate(cfg.suppliers)}
        self._operators = np.array(list(cfg.teams.values()))  # [team, line]
        self._shift_names = np.array(cfg.shifts.names)

        # Shift days start at day_start: find the last shift-day start at or before `start`.
        anchor = datetime.combine(self.start.date(), cfg.shifts.day_start)
        if anchor > self.start:
            anchor -= timedelta(days=1)
        self._anchor_date = anchor.date()
        self._offset = (self.start - anchor).total_seconds()

        dm = cfg.defect_model
        with np.errstate(divide="ignore"):
            self._log_base = np.log([dm.base_rate.get(c, 0.0) for c in self.classes])
        self._coef = np.array(
            [[dm.process.get(c, {}).get(p, 0.0) for p in self.params] for c in self.classes]
        )
        self._resin = np.array(
            [[dm.resin.get(c, {}).get(k, 0.0) for k in ("moisture", "contamination")] for c in self.classes]
        )
        self._allowed = {m.id: self._allowed_classes(m) for _, m in cfg.machines}
        self._warn_unsimulated()

        rng = np.random.default_rng([cfg.seed, 0])
        self._state = {
            m.id: _MachineState(
                next_shot=rng.uniform(0, m.process["cycle_time"]),
                ar={p: rng.normal(0, spec.drift_sd) for p, spec in cfg.process_params.items()},
            )
            for _, m in cfg.machines
        }

    # ------------------------------------------------------------------ setup helpers

    def _allowed_classes(self, m: Machine) -> np.ndarray:
        if self.pool is None:
            return np.ones(len(self.classes), bool)
        if m.part_type not in self.pool.categories:
            raise ValueError(
                f"{m.id} makes {m.part_type!r} but the image pool has no such category "
                f"(pool: {sorted(self.pool.categories)})"
            )
        available = self.pool.classes(m.part_type)
        if GOOD not in available:
            raise ValueError(f"image pool has no good samples of {m.part_type!r}")
        return np.array([c in available for c in self.classes])

    def _warn_unsimulated(self) -> None:
        base = dict(zip(self.classes, np.exp(self._log_base), strict=True))
        for _, m in self.cfg.machines:
            allowed = dict(zip(self.classes, self._allowed[m.id], strict=True))
            missing = [c for c in self.classes if base[c] > 0 and not allowed[c]]
            if missing:
                log.warning(
                    "%s: no %s images of %s; those defects are not simulated", m.id, missing, m.part_type
                )
        for f in self.cfg.faults:
            scope = [m for _, m in self.cfg.machines if f.machine in (None, m.id)]
            dead = [
                c
                for c in fault_targets(f, self.cfg)
                if not any(self._allowed[m.id][self.classes.index(c)] for m in scope)
            ]
            if dead:
                log.warning("fault %s targets %s, which have no images on the affected machines", f.id, dead)

    def lot(self, supplier: Supplier, index: int) -> _Lot:
        key = (supplier.id, index)
        if key not in self._lots:
            rng = np.random.default_rng([self.cfg.seed, 3, self._supplier_index[supplier.id], index])
            sd = self.cfg.defect_model.lot_sd
            moisture = supplier.moisture * math.exp(rng.normal(0, sd))
            contamination = supplier.contamination * math.exp(rng.normal(0, sd))
            self._lots[key] = _Lot(
                lot_id=supplier.lot_id(index),
                supplier_id=supplier.id,
                received=self.start + timedelta(days=index * supplier.lot_days),
                moisture_pct=round(0.08 * moisture + rng.normal(0, 0.004), 3),
                moisture=moisture,
                contamination=contamination,
            )
        return self._lots[key]

    def lots(self) -> pd.DataFrame:
        """Resin lots used so far, with their incoming-inspection data (hidden factors excluded)."""
        rows = [
            {
                "lot_id": x.lot_id,
                "supplier_id": x.supplier_id,
                "received": x.received,
                "moisture_pct": x.moisture_pct,
            }
            for x in self._lots.values()
        ]
        return pd.DataFrame(rows).sort_values("lot_id", ignore_index=True)

    # ------------------------------------------------------------------ generation

    def run(self, days: int | None = None) -> pd.DataFrame:
        return pd.concat([self.run_day() for _ in range(days or self.cfg.days)], ignore_index=True)

    def iter_parts(self, days: int | None = None) -> Iterator[dict]:
        """Part records in time order; endless when ``days`` is None."""
        d = 0
        while days is None or d < days:
            df = self.run_day().astype(object)
            yield from df.where(df.notna(), None).to_dict("records")  # missing values as None, not NaN
            d += 1

    def run_day(self) -> pd.DataFrame:
        frames = [self._machine_day(i, line.id, m) for i, (line, m) in enumerate(self.cfg.machines)]
        self.day += 1
        df = pd.concat([f for f in frames if f is not None], ignore_index=True)
        return df.sort_values(["timestamp", "machine_id", "cavity"], ignore_index=True)

    def _stops(self, m: Machine, rng: np.random.Generator, t0: float, t1: float) -> list[tuple[float, float]]:
        sp = self.cfg.stoppages
        k = rng.poisson(sp.per_day)
        starts = rng.uniform(t0, t1, k)
        ends = starts + np.maximum(rng.exponential(sp.mean_minutes * 60, k), 120)
        stops = [(s, e, "unplanned") for s, e in zip(starts, ends, strict=True)]
        for f in self.cfg.faults:
            end = f.to_day * DAY
            if f.kind == "drift" and f.machine == m.id and f.maintenance_hours and t0 <= end < t1:
                stops.append((end, end + f.maintenance_hours * 3600, f"maintenance ({f.id})"))
        merged: list[list] = []
        for s, e, reason in sorted(stops):
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, reason])
        for s, e, reason in merged:
            self.stoppages.append(
                {"machine_id": m.id, "start": self._ts(s), "end": self._ts(e), "reason": reason}
            )
        return [(s, e) for s, e, _ in merged]

    def _ts(self, seconds):
        return pd.Timestamp(self.start) + pd.to_timedelta(np.round(np.asarray(seconds) * 1000), unit="ms")

    def _machine_day(self, mi: int, line_id: str, m: Machine) -> pd.DataFrame | None:
        cfg, st = self.cfg, self._state[m.id]
        rng = np.random.default_rng([cfg.seed, 1, mi, self.day])
        t0, t1 = self.day * DAY, (self.day + 1) * DAY
        stops = self._stops(m, rng, t0, t1)

        # Process parameters per shot: setpoint + slow AR(1) drift + shot noise.
        c0 = m.process["cycle_time"]
        n = max(int((t1 - st.next_shot) / (0.5 * c0)) + 2, 2)
        values, drift = {}, {}
        for p, spec in cfg.process_params.items():
            phi = math.exp(-c0 / (spec.drift_hours * 3600))
            innov = rng.normal(0, spec.drift_sd * math.sqrt(1 - phi**2), n)
            drift[p], _ = lfilter([1.0], [1.0, -phi], innov, zi=[phi * st.ar[p]])
            values[p] = m.process[p] + drift[p] + rng.normal(0, spec.sd, n)
        cycle = values["cycle_time"] = np.maximum(values["cycle_time"], 0.5 * c0)

        # Shot times; a stop delays every later shot until the machine restarts.
        times = st.next_shot + np.concatenate([[0.0], np.cumsum(cycle[:-1])])
        for s, e in stops:
            j = np.searchsorted(times, s)
            if j < n and times[j] < e:
                times[j:] += e - times[j]
        k = int(np.searchsorted(times, t1))
        if k >= n:
            raise RuntimeError("shot buffer too small")
        st.next_shot = float(times[k])
        if k == 0:
            return None
        st.ar = {p: float(d[k - 1]) for p, d in drift.items()}
        times = times[:k]
        values = {p: v[:k] for p, v in values.items()}
        for f in cfg.faults:
            if f.kind == "drift" and f.machine == m.id:
                on = (times >= f.from_day * DAY) & (times < f.to_day * DAY)
                values[f.parameter][on] += f.per_day * (times[on] - f.from_day * DAY) / DAY
        shot_no = st.shot_no + 1 + np.arange(k)
        st.shot_no += k

        # One record per cavity.
        shot = np.repeat(np.arange(k), m.cavities)
        cavity = np.tile(np.arange(1, m.cavities + 1), k)
        t = times[shot]
        n_parts = len(t)
        u = t + self._offset
        day_idx = np.floor(u / DAY).astype(int)
        shift_idx = ((u - day_idx * DAY) // (cfg.shifts.hours * 3600)).astype(int)
        team_idx = (shift_idx - day_idx // 7) % len(cfg.teams)
        operator = self._operators[team_idx, (self._line_index[line_id] + day_idx) % len(cfg.lines)]
        supplier, lot, lot_factors = self._resin_supply(mi, m, t)

        # Defect probability per class, then one outcome per part.
        z = np.stack(
            [(values[p][shot] - m.process[p]) / cfg.process_params[p].tolerance for p in self.params], 1
        )
        logr = self._log_base + z @ self._coef.T + np.log(lot_factors) @ self._resin.T
        ctx = {
            "machine_id": m.id,
            "cavity": cavity,
            "operator_id": operator,
            "supplier_id": supplier,
            "resin_lot": lot,
            "shift": self._shift_names[shift_idx],
        }
        for f in cfg.faults:
            if f.kind == "multiplier" and (on := fault_scope(f, t, ctx)).any():
                for c, mult in f.classes.items():
                    logr[on, self.classes.index(c)] += math.log(mult)
        rate = np.exp(logr) * self._allowed[m.id]
        rate *= np.minimum(1.0, MAX_DEFECT_PROB / np.maximum(rate.sum(1), 1e-12))[:, None]
        outcome = (rng.random(n_parts)[:, None] >= np.cumsum(rate, 1)).sum(1)
        label = np.array([*self.classes, GOOD], dtype=object)[outcome]

        df = pd.DataFrame(
            {
                "part_id": [
                    f"ZD-{m.id.replace('-', '')}-{s:07d}-{c}"
                    for s, c in zip(shot_no[shot], cavity, strict=True)
                ],
                "timestamp": self._ts(t),
                "shift_date": [self._anchor_date + timedelta(days=int(d)) for d in day_idx],
                "shift": ctx["shift"],
                "line_id": line_id,
                "machine_id": m.id,
                "mould_id": m.mould,
                "cavity": cavity,
                "shot_no": shot_no[shot],
                "part_type": m.part_type,
                "material": m.material,
                "operator_id": operator,
                "supplier_id": supplier,
                "resin_lot": lot,
                **{p: np.round(values[p][shot], 2) for p in self.params},
                "true_label": np.where(label == GOOD, "good", "defect"),
                "true_class": np.where(label == GOOD, None, label),
            }
        )
        if self.pool is not None:
            image = np.empty(n_parts, dtype=object)
            for c in np.unique(label):
                sel = label == c
                image[sel] = self.pool.sample(m.part_type, c, int(sel.sum()), rng)
            df["image_sample_id"] = image
        return df

    def _resin_supply(self, mi: int, m: Machine, t: np.ndarray):
        """Supplier per resin window, then the supplier's lot current at each part's time."""
        cfg = self.cfg
        n = len(t)
        supplier = np.empty(n, dtype=object)
        window = np.floor(t / (cfg.resin_change_hours * 3600)).astype(int)
        names, shares = list(m.resin_mix), list(m.resin_mix.values())
        for w in np.unique(window):
            supplier[window == w] = np.random.default_rng([cfg.seed, 2, mi, w]).choice(names, p=shares)
        lot = np.empty(n, dtype=object)
        factors = np.empty((n, 2))
        for sid in np.unique(supplier):
            sup = cfg.supplier(sid)
            rows = np.flatnonzero(supplier == sid)
            index = np.floor(t[rows] / (sup.lot_days * DAY)).astype(int)
            for li in np.unique(index):
                x = self.lot(sup, int(li))
                sel = rows[index == li]
                lot[sel] = x.lot_id
                factors[sel] = (x.moisture, x.contamination)
        return supplier, lot, factors


def fault_effects(parts: pd.DataFrame, cfg: PlantConfig, start: datetime | None = None) -> pd.DataFrame:
    """Rate of each fault's target defects inside its scope vs. a control group.

    The control group is the same machine outside the scope for machine faults,
    and all other parts otherwise. This checks the faults were planted; it is
    not root-cause analysis (which must find them without this table).
    """
    start = pd.Timestamp(start or cfg.start)
    t = (parts.timestamp - start).dt.total_seconds().to_numpy()
    ctx = {
        c: parts[c].to_numpy()
        for c in ("machine_id", "cavity", "operator_id", "supplier_id", "resin_lot", "shift")
    }
    rows = []
    for f in cfg.faults:
        scope = fault_scope(f, t, ctx)
        control = ~scope & ((ctx["machine_id"] == f.machine) if f.machine else True)
        targets = fault_targets(f, cfg)
        hit = parts.true_class.isin(targets).to_numpy()
        rate_in = hit[scope].mean() if scope.any() else float("nan")
        rate_out = hit[control].mean() if control.any() else float("nan")
        rows.append(
            {
                "fault": f.id,
                "targets": ",".join(targets),
                "parts_in_scope": int(scope.sum()),
                "rate_in_scope": rate_in,
                "rate_control": rate_out,
                "lift": rate_in / rate_out if rate_out else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def save_run(out_dir, sim: ProductionSimulator, parts: pd.DataFrame) -> None:
    """Write a simulation run: parts, lots, stoppages, ground truth and the config used."""
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parts.to_parquet(out / "parts.parquet", index=False)
    sim.lots().to_csv(out / "lots.csv", index=False)
    pd.DataFrame(sim.stoppages, columns=["machine_id", "start", "end", "reason"]).to_csv(
        out / "stoppages.csv", index=False
    )
    effects = fault_effects(parts, sim.cfg, sim.start).set_index("fault")
    truth = {
        "note": "Production metadata is SIMULATED; images and their defects are real dataset samples.",
        "start": sim.start.isoformat(),
        "faults": [
            {
                **f.model_dump(mode="json", exclude_none=True),
                "start": (sim.start + timedelta(days=f.from_day)).isoformat(),
                "end": None if math.isinf(f.to_day) else (sim.start + timedelta(days=f.to_day)).isoformat(),
                "targets": fault_targets(f, sim.cfg),
                "observed": effects.loc[f.id].drop("targets").to_dict(),
            }
            for f in sim.cfg.faults
        ],
        "supplier_quality": [
            s.model_dump() for s in sim.cfg.suppliers if (s.moisture, s.contamination) != (1, 1)
        ],
    }
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2, default=str), encoding="utf-8")
    (out / "plant.json").write_text(sim.cfg.model_dump_json(indent=2), encoding="utf-8")
