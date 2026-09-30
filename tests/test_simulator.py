import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ml.datasets.common import load_manifests
from simulator.config import load_config
from simulator.images import ImagePool
from simulator.production import ProductionSimulator, fault_effects, save_run


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def history(cfg):
    """Metadata-only run of the full configured period (fast, no images)."""
    return ProductionSimulator(cfg).run()


def test_default_plant_matches_plan(cfg):
    assert [line.id for line in cfg.lines] == ["L1", "L2", "L3"]
    assert [m.id for _, m in cfg.machines] == [f"IMM-0{i}" for i in range(1, 7)]
    assert all(m.cavities == 2 for _, m in cfg.machines)
    assert len(cfg.operators) == 9 and len(cfg.suppliers) == 4
    assert cfg.shifts.names == ["A", "B", "C"]


def test_schema_and_ids(history):
    assert history.part_id.is_unique
    assert history.timestamp.is_monotonic_increasing
    assert set(history.true_label) == {"good", "defect"}
    assert (history.true_class.isna() == (history.true_label == "good")).all()
    assert set(history.cavity) == {1, 2}
    assert history.groupby(["machine_id", "shot_no"]).size().eq(2).all()


def test_shifts_and_operators(history):
    hour = history.timestamp.dt.hour
    expected = np.select([(hour >= 6) & (hour < 14), (hour >= 14) & (hour < 22)], ["A", "B"], "C")
    assert (history["shift"].to_numpy() == expected).all()
    # Night shift after midnight belongs to the previous shift date.
    night = history[(history["shift"] == "C") & (hour < 6)]
    assert (pd.to_datetime(night.shift_date) == night.timestamp.dt.normalize() - pd.Timedelta(days=1)).all()
    # Exactly one operator per line and shift, and everyone works.
    assert history.groupby(["shift_date", "shift", "line_id"]).operator_id.nunique().eq(1).all()
    assert history.operator_id.nunique() == 9


def test_resin_lots_follow_suppliers(history, cfg):
    assert (history.resin_lot.str.rsplit("-L", n=1).str[0] == history.supplier_id).all()
    for _, m in cfg.machines:
        used = set(history[history.machine_id == m.id].supplier_id)
        assert used <= set(m.resin_mix)


def test_planted_faults_are_detectable(history, cfg):
    effects = fault_effects(history, cfg).set_index("fault")
    assert list(effects.index) == ["F1", "F2", "F3", "F4"]
    assert (effects.parts_in_scope > 1000).all()
    assert (effects.lift > 1.8).all(), effects


def test_drift_fault_moves_parameter_and_recovers(history, cfg):
    m3 = history[history.machine_id == "IMM-03"]
    days = (m3.timestamp - pd.Timestamp(cfg.start)).dt.total_seconds() / 86400
    before = m3[days < 3].mould_temp.mean()
    late_drift = m3[(days > 7.5) & (days < 8.6)].mould_temp.mean()
    after = m3[days > 9].mould_temp.mean()
    assert before - late_drift > 8  # -2.5 degC/day for ~4.5 days
    assert abs(after - before) < 1.5


def test_deterministic_and_chunk_independent(cfg):
    a = ProductionSimulator(cfg).run(days=2)
    b = ProductionSimulator(cfg).run(days=2)
    pd.testing.assert_frame_equal(a, b)
    streamed = pd.DataFrame(list(ProductionSimulator(cfg).iter_parts(days=2)))
    assert streamed.part_id.tolist() == a.part_id.tolist()


def test_live_start_shifts_timeline(cfg):
    start = datetime(2030, 1, 1, 13, 30)
    df = ProductionSimulator(cfg, start=start).run(days=1)
    assert df.timestamp.min() >= pd.Timestamp(start)
    first = df.iloc[0]
    assert first["shift"] == "A" and str(first.shift_date) == "2030-01-01"


def test_image_binding_matches_class(demo_data, monkeypatch, cfg, tmp_path):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(demo_data))
    pool = ImagePool.from_sources(["demo"])
    sim = ProductionSimulator(cfg, pool)
    parts = sim.run(days=1)
    manifest = load_manifests(["demo"])
    sample_class = manifest.groupby("sample_id").zd_class.first().fillna("good")
    bound = sample_class.loc[parts.image_sample_id].to_numpy()
    assert (bound == parts.true_class.fillna("good").to_numpy()).all()
    assert set(manifest.set_index("sample_id").loc[parts.image_sample_id].split) == {"test"}
    assert len(pool.views[parts.image_sample_id.iloc[0]]) == 5

    save_run(tmp_path / "run", sim, parts)
    truth = json.loads((tmp_path / "run/ground_truth.json").read_text())
    assert [f["id"] for f in truth["faults"]] == ["F1", "F2", "F3", "F4"]
    assert "SIMULATED" in truth["note"]
    assert pd.read_parquet(tmp_path / "run/parts.parquet").shape == parts.shape


def test_classes_without_images_are_not_simulated(demo_data, monkeypatch, cfg):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(demo_data))
    manifest = load_manifests(["demo"])
    no_cracks = manifest[manifest.zd_class != "crack"]
    parts = ProductionSimulator(cfg, ImagePool(no_cracks)).run(days=1)
    assert "crack" not in set(parts.true_class.dropna())


def test_config_validation_reports_bad_references(cfg):
    data = cfg.model_dump()
    data["faults"][0]["machine"] = "IMM-99"
    data["lines"][0]["machines"][0]["resin_mix"] = {"SUP-Z": 1.0}
    with pytest.raises(ValueError) as e:
        type(cfg).model_validate(data)
    assert "IMM-99" in str(e.value) and "resin_mix" in str(e.value)
