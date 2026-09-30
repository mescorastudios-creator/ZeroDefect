"""Defect finding: PatchCore anomaly detection plus a nearest-neighbour defect-type classifier.

PatchCore (Roth et al., CVPR 2022) learns what GOOD parts look like from good images only
and flags regions whose local features have no close match in that memory, so it also
catches defect types it has never seen. A k-nearest-neighbour classifier over labelled
defect examples then names the defect type from the most anomalous patches.
Runs on a CPU: ResNet-18 features at 224 px.
"""

import base64
import logging
import math
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from scipy.stats import rankdata

from ml.datasets.common import instances_from_mask, load_manifests, resolve

log = logging.getLogger(__name__)

SIZE = 224
BANK_SIZE = 4000  # memory bank patches per part type (small enough to run in a browser)
TOP_PATCHES = 8
K_NEIGHBOURS = 5
MIN_VIEWS = 2  # a part fails when at least this many camera views show a defect (multi-view consistency)
GOOD_MARGIN = 1.1  # the threshold stays this far above every held-out good image (few calibration images)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare(im: np.ndarray) -> np.ndarray:
    """BGR image -> RGB float SIZE x SIZE. Large images are first averaged over k x k pixel blocks
    (k = whole-number downscale factor) so thin defects survive; the browser does the same steps."""
    x = im.astype(np.float32)
    k = min(im.shape[:2]) // SIZE
    if k >= 2:
        h, w = im.shape[0] // k * k, im.shape[1] // k * k
        x = x[:h, :w].reshape(h // k, k, w // k, k, 3).mean((1, 3))
    x = np.round(cv2.resize(x, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR))
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)


class Backbone:
    """ImageNet ResNet-18 layer2 + layer3 features, averaged over 3x3 neighbourhoods -> (B, 384, 28, 28)."""

    def __init__(self):
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT).eval()
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray]) -> torch.Tensor:
        out = []
        for i in range(0, len(images), 32):
            batch = [prepare(im) for im in images[i : i + 32]]
            x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).div(255).sub(_MEAN).div(_STD)
            f2 = self.layer2(self.stem(x))
            f3 = F.interpolate(self.layer3(f2), size=f2.shape[-2:], mode="bilinear", align_corners=False)
            out.append(F.avg_pool2d(torch.cat([f2, f3], 1), 3, 1, 1))
        return torch.cat(out)


def _coreset(patches: torch.Tensor, n: int, rng: np.random.Generator) -> torch.Tensor:
    """Greedy k-center coreset (PatchCore): n patches that cover all normal patches, rare ones included.

    Runs on a random projection of up to 10n candidate patches to stay fast.
    """
    if len(patches) <= n:
        return patches
    cand = patches[torch.from_numpy(rng.choice(len(patches), min(len(patches), 10 * n), replace=False))]
    proj = cand @ torch.from_numpy(rng.standard_normal((cand.shape[1], 128)).astype(np.float32))
    sel = [int(rng.integers(len(cand)))]
    dist = ((proj - proj[sel[0]]) ** 2).sum(1)
    for _ in range(n - 1):
        sel.append(int(dist.argmax()))
        dist = torch.minimum(dist, ((proj - proj[sel[-1]]) ** 2).sum(1))
    return cand[sel].clone()


def _best_threshold(neg: np.ndarray, pos: np.ndarray) -> float:
    """Score threshold with the best balanced accuracy between good (neg) and defective (pos) images."""
    s = np.sort(np.concatenate([neg, pos]))
    mids = (s[:-1] + s[1:]) / 2
    acc = [(np.mean(neg <= t) + np.mean(pos > t)) / 2 for t in mids]
    return float(mids[int(np.argmax(acc))])


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    pos = labels.astype(bool)
    ranks = rankdata(scores)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (
        float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else math.nan
    )


class PartModel:
    """Anomaly memory bank, decision threshold and defect-type examples for one part type."""

    def __init__(self, bank, threshold, class_vecs, class_labels, metrics=None):
        self.bank, self.threshold = bank, threshold
        self.class_vecs, self.class_labels = class_vecs, class_labels
        self.metrics = metrics or {}

    # ------------------------------------------------------------------ core
    def _maps(self, feats: torch.Tensor) -> torch.Tensor:
        """Patch anomaly scores: distance to the nearest good patch -> (B, 28, 28)."""
        b, c, h, w = feats.shape
        q = feats.permute(0, 2, 3, 1).reshape(-1, c)
        d = torch.cat([torch.cdist(q[i : i + 4096], self.bank).min(1).values for i in range(0, len(q), 4096)])
        return d.reshape(b, h, w)

    @staticmethod
    def _full(lowmap: torch.Tensor, shape) -> np.ndarray:
        h, w = shape[:2]
        m = cv2.resize(lowmap.numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), 4 * w / 256)

    @staticmethod
    def _vector(feat: torch.Tensor, lowmap: torch.Tensor) -> torch.Tensor:
        """Anomaly-weighted mean feature of the most anomalous patches, L2-normalised."""
        flat = lowmap.flatten()
        top = flat.topk(TOP_PATCHES).indices
        v = (feat.flatten(1)[:, top] * flat[top]).sum(1)
        return F.normalize(v, dim=0)

    def _classify(self, vec: torch.Tensor) -> tuple[str, float]:
        sims = self.class_vecs @ vec
        best = sims.topk(min(K_NEIGHBOURS, len(sims)))
        votes: dict[str, float] = {}
        for s, i in zip(best.values.tolist(), best.indices.tolist(), strict=True):
            votes[self.class_labels[i]] = votes.get(self.class_labels[i], 0.0) + max(s, 0.0)
        label = max(votes, key=votes.get)
        return label, votes[label] / (sum(votes.values()) or 1)

    def analyse(self, feats: torch.Tensor, images: list[np.ndarray]) -> list[dict]:
        """Per-view result: score, defect flag, predicted class, confidence, boxes (% of image), heat map."""
        lowmaps = self._maps(feats)
        out = []
        for feat, low, img in zip(feats, lowmaps, images, strict=True):
            hmap = self._full(low, img.shape)
            score = float(hmap.max())
            res = {
                "score": score,
                "defect": score > self.threshold,
                "cls": None,
                "confidence": None,
                "boxes": [],
                "heatmap": hmap,
            }
            if res["defect"]:
                res["cls"], res["confidence"] = self._classify(self._vector(feat, low))
                h, w = img.shape[:2]
                mask = (hmap > 0.75 * self.threshold).astype(np.uint8)
                for inst in instances_from_mask(mask, res["cls"], merge_px=max(5, w // 50)):
                    x, y, bw, bh = inst.bbox
                    if hmap[y : y + bh, x : x + bw].max() <= self.threshold:
                        continue  # only regions that actually cross the threshold
                    res["boxes"].append(
                        {
                            "x": 100 * x / w,
                            "y": 100 * y / h,
                            "w": 100 * bw / w,
                            "h": 100 * bh / h,
                            "cls": res["cls"],
                        }
                    )
            out.append(res)
        return out

    # ------------------------------------------------------------------ training
    @classmethod
    def fit(
        cls,
        backbone: Backbone,
        good: list[np.ndarray],
        defects: list[tuple[np.ndarray, str]],
        rng: np.random.Generator,
    ) -> "PartModel":
        idx = rng.permutation(len(good))
        n_cal = max(1, len(good) // 5)  # held-out good images to set the threshold
        cal, train = [good[i] for i in idx[:n_cal]], [good[i] for i in idx[n_cal:]]
        patches = backbone(train).permute(0, 2, 3, 1).reshape(-1, 384)
        model = cls(_coreset(patches, BANK_SIZE, rng), math.inf, None, None)

        def scores(images):
            feats = backbone(images)
            lows = model._maps(feats)
            return (
                feats,
                lows,
                np.array([model._full(lw, im.shape).max() for lw, im in zip(lows, images, strict=True)]),
            )

        _, _, s_good = scores(cal)
        d_imgs = [im for im, _ in defects]
        d_feats, d_lows, s_def = scores(d_imgs)
        model.threshold = max(_best_threshold(s_good, s_def), GOOD_MARGIN * float(s_good.max()))
        model.class_vecs = torch.stack([model._vector(f, lw) for f, lw in zip(d_feats, d_lows, strict=True)])
        model.class_labels = [label for _, label in defects]
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bank": self.bank,
                "threshold": self.threshold,
                "class_vecs": self.class_vecs,
                "class_labels": self.class_labels,
                "metrics": self.metrics,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "PartModel":
        return cls(**torch.load(path, weights_only=False))


def _read(relpath: str) -> np.ndarray:
    img = cv2.imread(str(resolve(relpath)), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(relpath)
    return img


def calibration_half(test: pd.DataFrame) -> pd.Series:
    """Defective test samples set aside (deterministically, by sample id) to calibrate a dataset
    that has no labelled defects in its train split; the rest stays held out for evaluation."""
    return (test.label == "defect") & test.sample_id.map(lambda s: zlib.crc32(s.encode()) % 2 == 0)


class Inspector:
    """One PartModel per part type, plus part-level (multi-view) decisions."""

    def __init__(self, models: dict[str, PartModel], backbone: Backbone):
        self.models, self.backbone = models, backbone

    @classmethod
    def load_or_fit(cls, sources, model_dir: Path, seed: int = 0) -> "Inspector":
        """Load the model of every part type in the given dataset sources, training missing ones."""
        backbone = Backbone()
        models = {}
        for source in [sources] if isinstance(sources, str) else sources:
            for cat, df in load_manifests([source]).groupby("category"):
                path = model_dir / f"{source}_{cat}.pt"
                if path.exists():
                    models[cat] = PartModel.load(path)
                    continue
                log.warning("training the defect model for %s (first run only)", cat)
                train, test = df[df.split == "train"], df[df.split == "test"]
                calib = train[train.label == "defect"]
                if calib.empty:  # e.g. MVTec AD: labelled defects only in test -> calibrate on half of them
                    calib, test = test[calibration_half(test)], test[~calibration_half(test)]
                good = [_read(p) for p in train[train.label == "good"].image_path]
                defects = [
                    (_read(p), c) for p, c in calib[["image_path", "zd_class"]].itertuples(index=False)
                ]
                if not good or not defects:
                    raise ValueError(f"{source}/{cat}: need good images and labelled defect images")
                model = PartModel.fit(backbone, good, defects, np.random.default_rng(seed))
                model.metrics = cls({cat: model}, backbone).evaluate(cat, test)
                model.save(path)
                models[cat] = model
        return cls(models, backbone)

    def inspect_part(self, part_type: str, images: list[np.ndarray]) -> dict:
        """Inspect all camera views of one part; FAIL when enough views agree on a defect."""
        model = self.models[part_type]
        views = model.analyse(self.backbone(images), images)
        flagged = [v for v in views if v["defect"]]
        defect = len(flagged) >= min(MIN_VIEWS, len(views))
        cls_, votes = None, {}
        if defect:
            for v in flagged:
                votes[v["cls"]] = votes.get(v["cls"], 0.0) + v["confidence"]
            cls_ = max(votes, key=votes.get)
        return {
            "defect": defect,
            "cls": cls_,
            "confidence": (votes[cls_] / sum(votes.values())) if defect else None,
            "score": max(v["score"] for v in views),
            "threshold": model.threshold,
            "flagged_views": len(flagged),
            "views": views,
        }

    def inspect_image(self, image: np.ndarray, part_type: str | None = None) -> dict:
        """Inspect one uploaded image; without a part type, use the model it fits best (lowest score)."""
        if part_type is None:
            feats = self.backbone([image])
            part_type = min(
                self.models, key=lambda k: float(self.models[k]._maps(feats).max()) / self.models[k].threshold
            )
        return {"part_type": part_type, **self.inspect_part(part_type, [image])}

    def evaluate(self, part_type: str, test: pd.DataFrame) -> dict:
        """Part-level and image-level results on the held-out test split."""
        view_labels, view_scores, rows = [], [], []
        for _, g in test.groupby("sample_id"):
            res = self.inspect_part(part_type, [_read(p) for p in g.image_path])
            truth = next((c for c in g.zd_class if isinstance(c, str)), None)
            rows.append((truth, res["defect"], res["cls"]))
            view_labels += list(g.label == "defect")
            view_scores += [v["score"] for v in res["views"]]
        truth = np.array([t is not None for t, _, _ in rows])
        pred = np.array([p for _, p, _ in rows])
        typed = [(t, c) for t, p, c in rows if t is not None and p]
        return {
            "parts": len(rows),
            "accuracy": float(np.mean(truth == pred)),
            "defects_found": float(pred[truth].mean()) if truth.any() else math.nan,
            "false_rejects": float(pred[~truth].mean()) if (~truth).any() else math.nan,
            "type_accuracy": float(np.mean([t == c for t, c in typed])) if typed else math.nan,
            "image_auroc": auroc(np.array(view_labels), np.array(view_scores)),
        }


def heatmap_png(hmap: np.ndarray, threshold: float, size: int = 320) -> str:
    """Transparent colour overlay of an anomaly map as a PNG data URI (clear where the part looks normal)."""
    m = cv2.resize(hmap, (size, round(size * hmap.shape[0] / hmap.shape[1])))
    level = np.clip((m - 0.8 * threshold) / (0.8 * threshold), 0, 1)  # 0 below 80% of the threshold
    color = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_JET)
    alpha = (np.clip(level * 3, 0, 1) * 190).astype(np.uint8)
    ok, buf = cv2.imencode(".png", np.dstack([color, alpha]))
    return "data:image/png;base64," + base64.b64encode(buf).decode()
