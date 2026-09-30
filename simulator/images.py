"""Pool of real dataset samples that simulated parts are bound to.

A *sample* is one physical part (all its camera views). Its class is the
defect class of its views, or ``good``. Samples come from the test split by
default, so a model trained on the train split never sees them.
"""

import numpy as np
import pandas as pd

from ml.datasets.common import load_manifests

GOOD = "good"


class ImagePool:
    def __init__(self, manifest: pd.DataFrame, split: str | None = "test"):
        df = manifest if split is None else manifest[manifest.split == split]
        if df.empty:
            raise ValueError(f"no samples in split {split!r}")
        df = df.assign(cls=df.zd_class.fillna(GOOD), sample_id=df.sample_id.fillna(df.image_id))
        # A sample is defective if any of its views shows a defect.
        per_sample = df.groupby(["category", "sample_id"]).cls.agg(
            lambda c: next((x for x in c if x != GOOD), GOOD)
        )
        groups: dict[tuple[str, str], list[str]] = {}
        for (category, sample_id), cls in per_sample.items():
            groups.setdefault((category, cls), []).append(sample_id)
        self._groups = {k: np.array(sorted(v)) for k, v in groups.items()}
        self.views = {
            sid: g.sort_values("view", na_position="first")[["view", "image_path", "mask_path"]].to_dict(
                "records"
            )
            for sid, g in df.groupby("sample_id")
        }
        self.sources = sorted(df.source.unique())

    @classmethod
    def from_sources(cls, sources, split: str | None = "test") -> "ImagePool":
        manifest = load_manifests(sources)
        if manifest.empty:
            raise FileNotFoundError(
                f"no converted samples for {list(sources)}; download/convert or run the demo"
            )
        return cls(manifest, split)

    @property
    def categories(self) -> set[str]:
        return {c for c, _ in self._groups}

    def classes(self, category: str) -> set[str]:
        """Classes (including ``good``) with at least one sample for the category."""
        return {cls for c, cls in self._groups if c == category}

    def sample(self, category: str, cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self._groups[(category, cls)], size=n)
