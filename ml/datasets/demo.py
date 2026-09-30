"""Procedural demo dataset in the unified format.

Simple drawings of the five Real-IAD polymer part types, 5 views per part,
with one drawn defect per defective part and a pixel mask. The train split has
good parts (for the anomaly model) and labelled defects (for the defect-type
classifier); the test split is held out for evaluation. It is NOT training
data: it exists so the simulator, virtual camera, tests and CI run without
downloading anything. Written to ``data/processed/demo/<part_type>``.
"""

from pathlib import Path

import cv2
import numpy as np

from ml.datasets.common import DEFECT, GOOD, Sample, processed_dir, rel, write_dataset

SOURCE = "demo"
PART_TYPES = ("plastic_nut", "plastic_plug", "end_cap", "u_block", "mounts")
DEFECTS = (
    "scratch",
    "crack",
    "deformation",
    "pit_void",
    "missing_material",
    "contamination",
    "discoloration",
)
_BG = 30.0
_TONE = {"plastic_nut": 150, "plastic_plug": 110, "end_cap": 170, "u_block": 130, "mounts": 95}


def _part_mask(part: str, s: int) -> np.ndarray:
    m = np.zeros((s, s), np.uint8)
    c = s // 2

    def pts(xy):
        return (np.array(xy) * s).astype(np.int32)

    if part == "plastic_nut":
        ang = np.deg2rad(np.arange(0, 360, 60) + 30)
        cv2.fillPoly(
            m, [np.stack([c + 0.32 * s * np.cos(ang), c + 0.32 * s * np.sin(ang)], 1).astype(np.int32)], 1
        )
        cv2.circle(m, (c, c), int(0.13 * s), 0, -1)
    elif part == "plastic_plug":
        cv2.circle(m, (c, c), int(0.3 * s), 1, -1)
        cv2.rectangle(m, (c, int(0.44 * s)), (int(0.9 * s), int(0.56 * s)), 1, -1)
    elif part == "end_cap":
        cv2.rectangle(m, (int(0.3 * s), int(0.3 * s)), (int(0.7 * s), int(0.7 * s)), 1, -1)
        for x in (0.3, 0.7):
            cv2.circle(m, (int(x * s), c), int(0.2 * s), 1, -1)
    elif part == "u_block":
        cv2.fillPoly(m, [pts([(0.2, 0.25), (0.8, 0.25), (0.8, 0.75), (0.2, 0.75)])], 1)
        cv2.fillPoly(m, [pts([(0.4, 0.25), (0.6, 0.25), (0.6, 0.58), (0.4, 0.58)])], 0)
    elif part == "mounts":
        cv2.rectangle(m, (int(0.12 * s), int(0.3 * s)), (int(0.88 * s), int(0.7 * s)), 1, -1)
        for x in (0.25, 0.75):
            cv2.circle(m, (int(x * s), c), int(0.07 * s), 0, -1)
    else:
        raise ValueError(part)
    return m


def _point(rng, part: np.ndarray, min_dist: float, max_dist: float = np.inf) -> tuple[int, int]:
    dist = cv2.distanceTransform(part, cv2.DIST_L2, 3)
    ys, xs = np.nonzero((dist >= min_dist) & (dist <= max(max_dist, min_dist + 1)))
    if not len(xs):  # tiny images: any point on the part
        ys, xs = np.nonzero(part)
    k = rng.integers(len(xs))
    return int(xs[k]), int(ys[k])


def _draw_defect(kind: str, img: np.ndarray, part: np.ndarray, rng, s: int) -> np.ndarray:
    """Draw one defect in place on ``img`` (float32 BGR); returns its mask."""
    m = np.zeros(part.shape, np.uint8)
    u = s / 256
    if kind == "scratch":
        x, y = _point(rng, part, 8 * u)
        a, n = rng.uniform(0, np.pi), rng.uniform(35, 70) * u
        p2 = (int(x + n * np.cos(a)), int(y + n * np.sin(a)))
        cv2.line(m, (x, y), p2, 1, max(1, int(1.5 * u)))
        m &= part
        img[m > 0] += 70
    elif kind == "crack":
        x, y = _point(rng, part, 1, 3 * u)
        pts = [(x, y)]
        cx, cy = s / 2, s / 2
        a = np.arctan2(cy - y, cx - x)
        for _ in range(8):
            a += rng.normal(0, 0.5)
            x, y = x + 6 * u * np.cos(a), y + 6 * u * np.sin(a)
            pts.append((int(x), int(y)))
        cv2.polylines(m, [np.array(pts, np.int32)], False, 1, max(1, int(1.5 * u)))
        m &= part
        img[m > 0] = 15
    elif kind == "deformation":
        x, y = _point(rng, part, 14 * u)
        axes = (int(rng.uniform(14, 24) * u), int(rng.uniform(7, 12) * u))
        cv2.ellipse(m, (x, y), axes, rng.uniform(0, 180), 0, 360, 1, -1)
        m &= part
        ramp = np.linspace(-45, 45, s, dtype=np.float32)[None, :].repeat(s, 0)
        img[m > 0] += ramp[m > 0][:, None]
    elif kind == "pit_void":
        for _ in range(rng.integers(1, 4)):
            x, y = _point(rng, part, 8 * u)
            cv2.circle(m, (x, y), max(2, int(rng.uniform(3, 6) * u)), 1, -1)
        img[m > 0] -= 85
    elif kind == "missing_material":
        x, y = _point(rng, part, 1, 2 * u)
        cv2.circle(m, (x, y), int(rng.uniform(20, 30) * u), 1, -1)
        m &= part
        img[m > 0] = _BG
    elif kind == "contamination":
        x0, y0 = _point(rng, part, 14 * u)
        for _ in range(rng.integers(4, 10)):
            x, y = x0 + int(rng.normal(0, 7 * u)), y0 + int(rng.normal(0, 7 * u))
            cv2.circle(m, (x, y), max(1, int(rng.uniform(1, 2.5) * u)), 1, -1)
        m &= part
        img[m > 0] = 10
    elif kind == "discoloration":
        x, y = _point(rng, part, 14 * u)
        alpha = np.zeros(part.shape, np.float32)
        cv2.ellipse(alpha, (x, y), (int(22 * u), int(14 * u)), rng.uniform(0, 180), 0, 360, 1.0, -1)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 5 * u) * part * 0.7
        img[:] = img * (1 - alpha[..., None]) + np.array([40, 90, 150], np.float32) * alpha[..., None]
        m = (alpha > 0.2).astype(np.uint8)
    else:
        raise ValueError(kind)
    return m


def _render(part_type: str, kind: str | None, rng, s: int) -> tuple[np.ndarray, np.ndarray]:
    part = _part_mask(part_type, s)
    tone = _TONE[part_type] + rng.normal(0, 4)
    img = np.full((s, s, 3), _BG, np.float32)
    yy, xx = np.mgrid[0:s, 0:s]
    shade = 1 - 0.25 * (((xx - s / 2) ** 2 + (yy - s / 2) ** 2) / (s / 2) ** 2)
    img[part > 0] = (tone * shade[part > 0])[:, None] * np.array([1.0, 1.0, 1.04], np.float32)
    mask = _draw_defect(kind, img, part, rng, s) if kind else np.zeros_like(part)
    return img, mask


def _view(img, mask, rng, s: int, v: int):
    angle = v * 72 + rng.normal(0, 4)
    mat = cv2.getRotationMatrix2D((s / 2, s / 2), angle, rng.uniform(0.96, 1.04))
    mat[:, 2] += rng.normal(0, 0.02 * s, 2)
    vi = cv2.warpAffine(img, mat, (s, s), flags=cv2.INTER_LINEAR, borderValue=(_BG,) * 3)
    vm = cv2.warpAffine(mask, mat, (s, s), flags=cv2.INTER_NEAREST)
    vi = vi * rng.uniform(0.85, 1.15) + rng.normal(0, 4, vi.shape)
    return np.clip(vi, 0, 255).astype(np.uint8), vm


def generate(
    part_types=PART_TYPES,
    size: int = 256,
    views: int = 5,
    train_good: int = 40,
    train_per_defect: int = 6,
    test_good: int = 20,
    test_per_defect: int = 4,
    seed: int = 0,
) -> list[Path]:
    outputs = []
    for ti, pt in enumerate(part_types):
        rng = np.random.default_rng([seed, ti])
        out = processed_dir(SOURCE, pt)
        plan = [("train", None)] * train_good + [
            ("train", d) for d in DEFECTS for _ in range(train_per_defect)
        ]
        plan += [("test", None)] * test_good
        plan += [("test", d) for d in DEFECTS for _ in range(test_per_defect)]
        samples = []
        for sid, (split, kind) in enumerate(plan, start=1):
            img, mask = _render(pt, kind, rng, size)
            label = kind or GOOD
            for v in range(1, views + 1):
                vi, vm = _view(img, mask, rng, size, v)
                stem = f"{pt}_{sid:04d}_{label}_C{v}"
                img_path = out / "images" / split / f"{stem}.jpg"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_path), vi, [cv2.IMWRITE_JPEG_QUALITY, 92])
                mask_path = None
                if kind:
                    mask_path = out / "masks" / split / f"{stem}.png"
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(mask_path), vm * 255)
                samples.append(
                    Sample(
                        image_id=f"{SOURCE}/{pt}/{stem}",
                        source=SOURCE,
                        category=pt,
                        split=split,
                        image_path=rel(img_path),
                        label=DEFECT if kind else GOOD,
                        zd_class=kind,
                        source_label=label,
                        mask_path=rel(mask_path) if mask_path else None,
                        sample_id=f"{SOURCE}/{pt}/{split}/S{sid:04d}",
                        view=f"C{v}",
                        width=size,
                        height=size,
                    )
                )
        outputs.append(write_dataset(out, samples))
    return outputs
