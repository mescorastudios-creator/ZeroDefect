"""Virtual camera: replays datasets, image folders or video streams at line speed.

Every source yields :class:`Capture` objects, one per inspected part (1..N
camera views) or per video frame, each with a timestamp. :func:`paced`
releases them in real time, optionally faster (``speed``).

    python -m simulator.camera --source sim --images demo --speed 60 --limit 20 --out outputs/preview
    python -m simulator.camera --source folder:data/raw/mvtec_ad/screw/test --interval 0.5
    python -m simulator.camera --source video:0            # webcam; also a file path or rtsp:// URL
"""

import argparse
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from ml.datasets.common import resolve

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_VIEW_RE = re.compile(r"_C(\d+)(?=[_.])")


@dataclass
class Frame:
    image: np.ndarray  # BGR uint8
    view: str | None = None
    path: str | None = None


@dataclass
class Capture:
    capture_id: str
    timestamp: datetime
    frames: list[Frame]
    meta: dict = field(default_factory=dict)  # traceability record for simulated parts


def _read(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def simulated_parts(sim, days: int | None = None) -> Iterator[Capture]:
    """Parts from a :class:`~simulator.production.ProductionSimulator` with an image pool."""
    if sim.pool is None:
        raise ValueError("the simulator needs an image pool to produce camera frames")
    for rec in sim.iter_parts(days):
        views = sim.pool.views[rec["image_sample_id"]]
        frames = [Frame(_read(resolve(v["image_path"])), v["view"], v["image_path"]) for v in views]
        yield Capture(rec["part_id"], rec["timestamp"].to_pydatetime(), frames, rec)


def image_folder(
    folder: Path,
    interval: float = 1.0,
    loop: bool = False,
    group_views: bool = True,
    start: datetime | None = None,
) -> Iterator[Capture]:
    """Images from a folder tree, ``interval`` seconds apart.

    With ``group_views``, files named like ``<part>_C<n>_...`` (Real-IAD style)
    are grouped into one multi-view capture.
    """
    files = sorted(
        p
        for p in Path(folder).rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.endswith("_mask")
    )
    if not files:
        raise FileNotFoundError(f"no images under {folder}")
    groups: dict[str, list[Path]] = {}
    for p in files:
        key = str(p.parent / _VIEW_RE.split(p.name)[0]) if group_views and _VIEW_RE.search(p.name) else str(p)
        groups.setdefault(key, []).append(p)
    t = start or datetime.now()
    while True:
        for key, paths in groups.items():
            frames = [
                Frame(_read(p), f"C{m[1]}" if (m := _VIEW_RE.search(p.name)) else None, str(p)) for p in paths
            ]
            yield Capture(Path(key).name, t, frames)
            t += timedelta(seconds=interval)
        if not loop:
            return


def video(uri: str | int, start: datetime | None = None) -> Iterator[Capture]:
    """Frames from a video file, webcam index or stream URL (rtsp://, http://)."""
    live = isinstance(uri, int) or "://" in str(uri)
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        raise OSError(f"cannot open video source {uri!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    t0 = start or datetime.now()
    i = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                return
            ts = datetime.now() if live else t0 + timedelta(seconds=i / fps)
            yield Capture(f"frame-{i:06d}", ts, [Frame(img)])
            i += 1
    finally:
        cap.release()


def paced(
    captures: Iterable[Capture],
    speed: float | None = 1.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[Capture]:
    """Release captures at their timestamps, ``speed`` times faster; ``None`` = as fast as possible.

    If reading falls behind, captures are released immediately until caught up.
    """
    t0 = wall0 = None
    for cap in captures:
        if speed:
            if t0 is None:
                t0, wall0 = cap.timestamp, clock()
            wait = wall0 + (cap.timestamp - t0).total_seconds() / speed - clock()
            if wait > 0:
                sleep(wait)
        yield cap


def contact_sheet(cap: Capture, height: int = 256) -> np.ndarray:
    """All views side by side with the part id and context, for previews."""
    tiles = [
        cv2.resize(f.image, (round(f.image.shape[1] * height / f.image.shape[0]), height)) for f in cap.frames
    ]
    sheet = np.hstack(tiles)
    meta = cap.meta
    text = (
        cap.capture_id
        if not meta
        else (
            f"{cap.capture_id}  {meta['machine_id']} cav{meta['cavity']}  {meta['operator_id']}  "
            f"shift {meta['shift']}  {meta['true_class'] or 'good'}"
        )
    )
    cv2.rectangle(sheet, (0, 0), (sheet.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(sheet, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return sheet


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m simulator.camera",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default="sim", help="sim | folder:<path> | video:<file|index|url>")
    p.add_argument("--images", default="demo", help="dataset sources for --source sim (default: demo)")
    p.add_argument(
        "--live", action="store_true", help="start the simulated timeline now instead of the config start"
    )
    p.add_argument("--speed", type=float, default=1.0, help="replay speed factor; 0 = as fast as possible")
    p.add_argument("--interval", type=float, default=1.0, help="seconds between folder images")
    p.add_argument("--limit", type=int, default=20, help="stop after this many captures; 0 = endless")
    p.add_argument("--out", type=Path, help="write a contact-sheet JPEG per capture here")
    args = p.parse_args(argv)

    if args.source == "sim":
        from simulator.config import load_config
        from simulator.images import ImagePool
        from simulator.production import ProductionSimulator

        start = datetime.now().replace(microsecond=0) if args.live else None
        sim = ProductionSimulator(load_config(), ImagePool.from_sources(args.images.split(",")), start=start)
        source = simulated_parts(sim)
    elif args.source.startswith("folder:"):
        source = image_folder(Path(args.source[7:]), interval=args.interval)
    elif args.source.startswith("video:"):
        uri = args.source[6:]
        source = video(int(uri) if uri.isdigit() else uri)
    else:
        p.error("--source must be sim, folder:<path> or video:<uri>")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for n, cap in enumerate(paced(source, args.speed or None), start=1):
        m = cap.meta
        ctx = f"  {m['machine_id']}  {m['true_class'] or 'good'}" if m else ""
        print(f"{cap.timestamp:%Y-%m-%d %H:%M:%S.%f}"[:-3], cap.capture_id, f"{len(cap.frames)} view(s){ctx}")
        if args.out:
            cv2.imwrite(str(args.out / f"{n:05d}_{cap.capture_id}.jpg"), contact_sheet(cap))
        if args.limit and n >= args.limit:
            break


if __name__ == "__main__":
    main()
