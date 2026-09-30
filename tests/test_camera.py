from datetime import datetime, timedelta

import cv2
import numpy as np
from conftest import write_image, write_mask

from simulator.camera import Capture, contact_sheet, image_folder, paced, simulated_parts, video
from simulator.config import load_config
from simulator.images import ImagePool
from simulator.production import ProductionSimulator

T0 = datetime(2026, 9, 1, 6)


def test_paced_follows_timestamps_at_speed():
    now = [100.0]
    sleeps = []

    def sleep(s):
        sleeps.append(round(s, 6))
        now[0] += s

    caps = [Capture(str(i), T0 + timedelta(seconds=s), []) for i, s in enumerate([0, 1, 3, 3])]
    out = list(paced(caps, speed=2, clock=lambda: now[0], sleep=sleep))
    assert [c.capture_id for c in out] == ["0", "1", "2", "3"]
    assert sleeps == [0.5, 1.0]


def test_paced_catches_up_when_behind():
    now = [0.0]
    sleeps = []

    def clock():
        now[0] += 10  # every read of the clock costs 10 s: always behind
        return now[0]

    caps = [Capture(str(i), T0 + timedelta(seconds=i), []) for i in range(3)]
    assert len(list(paced(caps, speed=1, clock=clock, sleep=sleeps.append))) == 3
    assert sleeps == []


def test_image_folder_groups_views_and_skips_masks(tmp_path):
    for view in (1, 2, 3):
        write_image(tmp_path / f"S0001/nut_0001_OK_C{view}_20231021.jpg")
    write_image(tmp_path / "S0002/nut_0002_NG_AK_C1_20231021.jpg")
    write_mask(tmp_path / "S0002/nut_0002_NG_AK_C1_20231021_mask.png", [(0, 0, 3, 3)])
    write_image(tmp_path / "single.png")
    caps = list(image_folder(tmp_path, interval=0.5, start=T0))
    assert [len(c.frames) for c in caps] == [3, 1, 1]
    assert [f.view for f in caps[0].frames] == ["C1", "C2", "C3"]
    assert caps[2].timestamp - caps[0].timestamp == timedelta(seconds=1)


def test_image_folder_loops(tmp_path):
    write_image(tmp_path / "a.png")
    gen = image_folder(tmp_path, loop=True, start=T0)
    assert [next(gen).capture_id for _ in range(3)] == ["a.png"] * 3


def test_video_file(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 24))
    for i in range(5):
        writer.write(np.full((24, 32, 3), i * 40, np.uint8))
    writer.release()
    caps = list(video(path, start=T0))
    assert len(caps) == 5
    assert caps[0].frames[0].image.shape == (24, 32, 3)
    assert caps[4].timestamp - caps[0].timestamp == timedelta(seconds=0.4)


def test_simulated_parts(demo_data, monkeypatch):
    monkeypatch.setenv("ZERODEFECT_DATA_ROOT", str(demo_data))
    sim = ProductionSimulator(load_config(), ImagePool.from_sources(["demo"]))
    caps = []
    for cap in simulated_parts(sim):
        caps.append(cap)
        if len(caps) == 10:
            break
    assert all(len(c.frames) == 5 and c.frames[0].image.shape == (64, 64, 3) for c in caps)
    assert caps[0].capture_id == caps[0].meta["part_id"]
    assert caps[0].meta["true_class"] is None or isinstance(caps[0].meta["true_class"], str)
    assert contact_sheet(caps[0], height=64).shape == (64, 5 * 64, 3)
