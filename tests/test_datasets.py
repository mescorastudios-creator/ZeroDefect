import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from conftest import write_image, write_mask

from ml.datasets import mvtec_ad, mvtec_ad2, paintdefect, realiad
from ml.datasets.common import extract_classes_from_tar, instances_from_mask, load_manifests


def _coco(path):
    return json.loads((path / "coco.json").read_text())


# --------------------------------------------------------------------------- masks


def test_instances_from_mask_groups_nearby_regions():
    m = np.zeros((40, 40), np.uint8)
    m[5:8, 5:8] = 1
    m[5:8, 10:13] = 1  # 2 px gap: same defect
    m[30:35, 30:35] = 1  # far away: separate defect
    insts = sorted(instances_from_mask(m, "scratch"), key=lambda i: i.bbox[0])
    assert [i.bbox for i in insts] == [[5, 5, 8, 3], [30, 30, 5, 5]]
    assert [i.area for i in insts] == [18, 25]
    assert all(len(poly) >= 6 for i in insts for poly in i.segmentation)


def test_instances_from_mask_drops_specks():
    m = np.zeros((20, 20), np.uint8)
    m[3, 3] = 1
    assert instances_from_mask(m, "pit_void", min_area=4) == []


# --------------------------------------------------------------------------- Real-IAD


def _realiad_fixture(data_root, with_json=True):
    cat = data_root / "raw/realiad/realiad_512/plastic_nut"
    entries = {"train": [], "test": []}
    for sid, status, code in [(1, "OK", "OK"), (2, "OK", "OK"), (3, "NG", "AK")]:
        folder = f"OK/S{sid:04d}" if status == "OK" else f"NG/{code}/S{sid:04d}"
        for view in (1, 2):
            stem = f"plastic_nut_{sid:04d}_{status}{'' if status == 'OK' else '_' + code}_C{view}_20231021"
            write_image(cat / folder / f"{stem}.jpg")
            mask = None
            if status == "NG":
                write_mask(cat / folder / f"{stem}.png", [(4, 4, 9, 9)])
                mask = f"{folder}/{stem}.png"
            split = "train" if sid == 1 else "test"
            entries[split].append(
                {"image_path": f"{folder}/{stem}.jpg", "mask_path": mask, "anomaly_class": code}
            )
    if with_json:
        js = data_root / "raw/realiad/realiad_jsons/realiad_jsons/plastic_nut.json"
        js.parent.mkdir(parents=True)
        js.write_text(json.dumps({"meta": {}, **entries}))


def test_realiad_convert_with_split_json(data_root):
    _realiad_fixture(data_root)
    (out,) = realiad.convert(classes=["plastic_nut"])
    df = load_manifests()
    assert len(df) == 6
    assert df.groupby("split").size().to_dict() == {"test": 4, "train": 2}
    ng = df[df.label == "defect"]
    assert set(ng.zd_class) == {"pit_void"} and set(ng.zd_subtype) == {"pit"}
    assert set(ng.source_label) == {"AK"}
    assert set(ng.view) == {"C1", "C2"}
    assert ng.sample_id.nunique() == 1 and ng.sample_id.iloc[0] == "realiad/plastic_nut/NG/AK/S0003"
    assert df.image_path.str.startswith("raw/realiad/realiad_512/plastic_nut/").all()
    coco = _coco(out)
    assert len(coco["images"]) == 6 and len(coco["annotations"]) == 2
    assert coco["annotations"][0]["bbox"] == [4, 4, 6, 6]
    assert coco["annotations"][0]["category_id"] == 4  # pit_void


def test_realiad_convert_without_split_json(data_root):
    _realiad_fixture(data_root, with_json=False)
    realiad.convert(classes=["plastic_nut"])
    df = load_manifests()
    assert len(df) == 6  # masks are not mistaken for images
    assert (df[df.label == "defect"].split == "test").all()
    assert df[df.label == "defect"].mask_path.notna().all()


def test_realiad_convert_requires_download(data_root):
    with pytest.raises(FileNotFoundError, match="download"):
        realiad.convert(classes=["plastic_nut"])


# --------------------------------------------------------------------------- MVTec AD / AD 2


def test_mvtec_ad_convert(data_root):
    root = data_root / "raw/mvtec_ad/screw"
    write_image(root / "train/good/000.png")
    write_image(root / "test/good/000.png")
    write_image(root / "test/scratch_head/000.png")
    write_mask(root / "ground_truth/scratch_head/000_mask.png", [(2, 2, 5, 20)])
    (out,) = mvtec_ad.convert(classes=["screw"])
    df = load_manifests().set_index("image_id")
    assert len(df) == 3 and df.index.is_unique
    row = df.loc["mvtec_ad/screw/test/scratch_head/000"]
    assert (row.label, row.zd_class, row.split) == ("defect", "scratch", "test")
    assert row.mask_path == "raw/mvtec_ad/screw/ground_truth/scratch_head/000_mask.png"
    assert [a["bbox"] for a in _coco(out)["annotations"]] == [[2, 2, 4, 19]]


def test_mvtec_ad2_convert(data_root):
    root = data_root / "raw/mvtec_ad2/wallplugs"
    write_image(root / "train/good/000_regular.png")
    write_image(root / "validation/good/000_regular.png")
    write_image(root / "test_public/good/000_regular.png")
    write_image(root / "test_public/bad/000_regular.png")
    write_mask(root / "test_public/ground_truth/bad/000_regular_mask.png", [(0, 0, 3, 3)])
    write_image(root / "test_private/000.png")  # no ground truth: not converted
    mvtec_ad2.convert()
    df = load_manifests()
    assert df.groupby("split").size().to_dict() == {"test": 2, "train": 1, "val": 1}
    bad = df[df.label == "defect"].iloc[0]
    assert bad.zd_class == "unknown" and bad.mask_path.endswith("000_regular_mask.png")


# --------------------------------------------------------------------------- PaintDefect


def test_paintdefect_convert(data_root):
    folder = data_root / "raw/paintdefect/v2/train"
    write_image(folder / "a.jpg", size=(50, 40))
    write_image(folder / "b.jpg", size=(50, 40))
    coco = {
        "categories": [
            {"id": 0, "name": "paint-defects"},
            {"id": 1, "name": "Scratch"},
            {"id": 2, "name": "dust"},
        ],
        "images": [
            {"id": 0, "file_name": "a.jpg", "width": 50, "height": 40},
            {"id": 1, "file_name": "b.jpg", "width": 50, "height": 40},
        ],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 2, "bbox": [1, 1, 5, 5], "area": 25},
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [20, 20, 4, 4], "area": 16},
            {"id": 2, "image_id": 0, "category_id": 1, "bbox": [10, 2, 3, 30], "area": 90},
        ],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    (out,) = paintdefect.convert()
    df = load_manifests().set_index("image_id")
    a = df.loc["paintdefect/painted_panel/train/a"]
    assert (a.label, a.zd_class, a.zd_subtype, a.source_label) == ("defect", "contamination", "dust", "dust")
    assert df.loc["paintdefect/painted_panel/train/b"].label == "good"
    cats = sorted(x["category_id"] for x in _coco(out)["annotations"])
    assert cats == [1, 6, 6]


def test_paintdefect_augmented_copies_are_one_part(data_root):
    folder = data_root / "raw/paintdefect/v2/train"
    names = ["p_jpg.rf.aaa.jpg", "p_jpg.rf.bbb.jpg", "q_jpg.rf.ccc.jpg"]
    for n in names:
        write_image(folder / n)
    coco = {
        "categories": [{"id": 1, "name": "sagging"}],
        "images": [{"id": i, "file_name": n, "width": 32, "height": 32} for i, n in enumerate(names)],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [1, 1, 5, 5]}],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    paintdefect.convert()
    df = load_manifests().set_index("image_id")
    parts = df["sample_id"].str.rsplit("/", n=1).str[-1]
    assert parts.to_dict() == {
        "paintdefect/painted_panel/train/p_jpg.rf.aaa": "p_jpg",
        "paintdefect/painted_panel/train/p_jpg.rf.bbb": "p_jpg",
        "paintdefect/painted_panel/train/q_jpg.rf.ccc": "q_jpg",
    }
    assert df.loc["paintdefect/painted_panel/train/p_jpg.rf.aaa"].zd_class == "paint_finish"


def test_paintdefect_unmapped_label_fails(data_root):
    folder = data_root / "raw/paintdefect/v1/train"
    write_image(folder / "a.jpg")
    coco = {
        "categories": [{"id": 1, "name": "orange_peel"}],
        "images": [{"id": 0, "file_name": "a.jpg"}],
        "annotations": [{"id": 0, "image_id": 0, "category_id": 1, "bbox": [0, 0, 2, 2]}],
    }
    (folder / "_annotations.coco.json").write_text(json.dumps(coco))
    with pytest.raises(KeyError, match="orange_peel"):
        paintdefect.convert()


# --------------------------------------------------------------------------- streaming tar extraction


def _tar_xz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def flaky_server():
    """HTTP server with Range support that cuts the first response short."""
    state = {"payload": b"", "cut": True, "fail_resume": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-")[6:].split("-")[0])
            state["requests"].append(start)
            if start and state["fail_resume"]:
                state["fail_resume"] -= 1
                self.send_error(503)
                return
            body = state["payload"][start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["cut"]:
                state["cut"] = False
                body = body[: len(body) // 2]
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_port}/archive.tar.xz"
    server.shutdown()


def test_extract_classes_from_tar_resumes_and_filters(flaky_server, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setattr("ml.datasets.common.time.sleep", lambda s: None)
    state, url = flaky_server
    state["fail_resume"] = 1  # the first reconnect fails too
    rng = np.random.default_rng(0)
    state["payload"] = _tar_xz(
        {
            "mvtec/bottle/train/good/000.png": rng.bytes(50_000),
            "mvtec/screw/train/good/000.png": rng.bytes(50_000),
            "mvtec/screw/license.txt": b"cc",
            "mvtec/tile/train/good/000.png": rng.bytes(50_000),
        }
    )
    extract_classes_from_tar(url, tmp_path, ["screw"])
    files = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file())
    assert files == ["screw/license.txt", "screw/train/good/000.png"]
    # cut short, one failed reconnect, then resumed with a Range request
    assert len(state["requests"]) == 3 and state["requests"][1] == state["requests"][2] > 0


def test_extract_classes_from_tar_reports_missing(flaky_server, tmp_path, monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    state, url = flaky_server
    state["cut"] = False
    state["payload"] = _tar_xz({"bottle/a.png": b"x"})
    with pytest.raises(FileNotFoundError, match="screw"):
        extract_classes_from_tar(url, tmp_path, ["screw"])
