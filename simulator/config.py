"""Plant and simulation settings, loaded from configs/plant.yaml."""

import math
import re
from datetime import datetime, time
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from inspection.paths import CONFIG_DIR
from inspection.taxonomy import load_taxonomy

LOT_RE = re.compile(r"^(?P<supplier>.+)-L(?P<index>\d+)$")


class Shifts(BaseModel):
    day_start: time = time(6)
    names: list[str] = ["A", "B", "C"]

    @property
    def hours(self) -> float:
        return 24 / len(self.names)


class Supplier(BaseModel):
    id: str
    name: str
    lot_days: float = Field(2, gt=0)
    moisture: float = Field(1.0, gt=0)  # mean lot factor (1 = nominal)
    contamination: float = Field(1.0, gt=0)

    def lot_id(self, index: int) -> str:
        return f"{self.id}-L{index + 1:03d}"


class Machine(BaseModel):
    id: str
    part_type: str
    material: str
    mould: str
    cavities: int = Field(2, ge=1)
    resin_mix: dict[str, float]
    process: dict[str, float]


class Line(BaseModel):
    id: str
    name: str
    machines: list[Machine]


class ProcessParam(BaseModel):
    unit: str
    sd: float = Field(ge=0)
    drift_sd: float = Field(ge=0)
    drift_hours: float = Field(gt=0)
    tolerance: float = Field(gt=0)


class DefectModel(BaseModel):
    base_rate: dict[str, float]
    process: dict[str, dict[str, float]] = {}
    resin: dict[str, dict[Literal["moisture", "contamination"], float]] = {}
    lot_sd: float = Field(0.15, ge=0)


class Stoppages(BaseModel):
    per_day: float = Field(1.0, ge=0)
    mean_minutes: float = Field(20, gt=0)


class Fault(BaseModel):
    id: str
    kind: Literal["drift", "multiplier"]
    description: str
    from_day: float = 0
    to_day: float = math.inf
    # Scope: a part is affected when every field that is set matches it.
    machine: str | None = None
    cavity: int | None = None
    operator: str | None = None
    supplier: str | None = None
    lot: str | None = None
    shift: str | None = None
    # drift: parameter offset grows linearly by `per_day`; maintenance stop at the end
    parameter: str | None = None
    per_day: float | None = None
    maintenance_hours: float = 0
    # multiplier: defect-rate multiplier per class
    classes: dict[str, float] = {}
    expected_root_cause: dict = {}

    @model_validator(mode="after")
    def _check_kind(self):
        if self.kind == "drift" and not (self.machine and self.parameter and self.per_day is not None):
            raise ValueError(f"{self.id}: a drift fault needs machine, parameter and per_day")
        if self.kind == "multiplier" and not self.classes:
            raise ValueError(f"{self.id}: a multiplier fault needs classes")
        if self.from_day >= self.to_day:
            raise ValueError(f"{self.id}: from_day must be before to_day")
        return self


class PlantConfig(BaseModel):
    seed: int = 0
    start: datetime
    days: int = Field(14, ge=1)
    shifts: Shifts = Shifts()
    teams: dict[str, list[str]]
    resin_change_hours: float = Field(8, gt=0)
    suppliers: list[Supplier]
    lines: list[Line]
    process_params: dict[str, ProcessParam]
    defect_model: DefectModel
    stoppages: Stoppages = Stoppages()
    faults: list[Fault] = []

    @property
    def machines(self) -> list[tuple[Line, Machine]]:
        return [(line, m) for line in self.lines for m in line.machines]

    @property
    def operators(self) -> list[str]:
        return [op for team in self.teams.values() for op in team]

    def supplier(self, supplier_id: str) -> Supplier:
        return next(s for s in self.suppliers if s.id == supplier_id)

    @model_validator(mode="after")
    def _check_references(self):
        classes = set(load_taxonomy().keys)
        params = set(self.process_params)
        suppliers = {s.id for s in self.suppliers}
        machines = {m.id: m for _, m in self.machines}
        errors = []
        if len(machines) != sum(len(line.machines) for line in self.lines):
            errors.append("machine ids must be unique")
        if 24 % len(self.shifts.names):
            errors.append("the number of shifts must divide 24 hours")
        if len(self.teams) != len(self.shifts.names):
            errors.append("need one operator team per shift")
        if any(len(t) != len(self.lines) for t in self.teams.values()):
            errors.append("each team needs one operator per line")
        if len(set(self.operators)) != len(self.operators):
            errors.append("operator ids must be unique")
        if "cycle_time" not in params:
            errors.append("process_params must include cycle_time")
        for m in machines.values():
            if set(m.process) != params:
                errors.append(f"{m.id}: process setpoints must be exactly {sorted(params)}")
            if not set(m.resin_mix) <= suppliers or not math.isclose(sum(m.resin_mix.values()), 1):
                errors.append(f"{m.id}: resin_mix must use known suppliers and sum to 1")
        dm = self.defect_model
        for cls in [*dm.base_rate, *dm.process, *dm.resin]:
            if cls not in classes:
                errors.append(f"defect_model: unknown class {cls!r}")
        for cls, coefs in dm.process.items():
            if not set(coefs) <= params:
                errors.append(f"defect_model.process.{cls}: unknown parameter in {sorted(coefs)}")
        for f in self.faults:
            if f.machine and f.machine not in machines:
                errors.append(f"{f.id}: unknown machine {f.machine}")
            if f.cavity and (not f.machine or f.cavity > machines[f.machine].cavities):
                errors.append(f"{f.id}: cavity needs a machine with that many cavities")
            if f.operator and f.operator not in self.operators:
                errors.append(f"{f.id}: unknown operator {f.operator}")
            if f.supplier and f.supplier not in suppliers:
                errors.append(f"{f.id}: unknown supplier {f.supplier}")
            if f.lot and (not (m := LOT_RE.match(f.lot)) or m["supplier"] not in suppliers):
                errors.append(f"{f.id}: lot must look like <supplier>-L<number>")
            if f.shift and f.shift not in self.shifts.names:
                errors.append(f"{f.id}: unknown shift {f.shift}")
            if f.parameter and f.parameter not in params - {"cycle_time"}:
                errors.append(f"{f.id}: drift parameter must be a process parameter other than cycle_time")
            if not set(f.classes) <= classes:
                errors.append(f"{f.id}: unknown class in {sorted(f.classes)}")
        if errors:
            raise ValueError("invalid plant config:\n  " + "\n  ".join(errors))
        return self


def load_config(path: Path | None = None) -> PlantConfig:
    path = path or CONFIG_DIR / "plant.yaml"
    with open(path, encoding="utf-8") as f:
        return PlantConfig.model_validate(yaml.safe_load(f))
