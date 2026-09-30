"""Unified defect taxonomy, loaded from configs/taxonomy.yaml.

Every dataset label is mapped to one ZeroDefect class (plus an optional
sub-type) through :meth:`Taxonomy.map_label`.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from inspection.paths import CONFIG_DIR

GOOD = "good"


@dataclass(frozen=True)
class DefectClass:
    id: int
    key: str
    name: str
    subtypes: tuple[str, ...]
    default_severity: str
    color: str


@dataclass(frozen=True)
class LabelMapping:
    """Result of mapping a source label: ``zd_class`` is None for good parts."""

    zd_class: str | None
    subtype: str | None = None

    @property
    def is_good(self) -> bool:
        return self.zd_class is None


class Taxonomy:
    def __init__(self, data: dict):
        self.version: int = data["version"]
        self.severity_levels: tuple[str, ...] = tuple(data["severity_levels"])
        self.classes: tuple[DefectClass, ...] = tuple(
            DefectClass(
                id=c["id"],
                key=c["key"],
                name=c["name"],
                subtypes=tuple(c.get("subtypes", [])),
                default_severity=c["default_severity"],
                color=c["color"],
            )
            for c in data["classes"]
        )
        self._by_key = {c.key: c for c in self.classes}
        self._by_id = {c.id: c for c in self.classes}
        if len(self._by_key) != len(self.classes) or len(self._by_id) != len(self.classes):
            raise ValueError("taxonomy class keys and ids must be unique")
        for c in self.classes:
            if c.default_severity not in self.severity_levels:
                raise ValueError(f"{c.key}: unknown severity {c.default_severity!r}")

        self._sources: dict[str, dict[str, LabelMapping]] = {}
        for source, labels in data["sources"].items():
            self._sources[source] = {
                str(label).lower(): self._parse_mapping(source, label, value)
                for label, value in labels.items()
            }

    def _parse_mapping(self, source: str, label, value) -> LabelMapping:
        if value == GOOD:
            return LabelMapping(None)
        key, subtype = (value["class"], value.get("subtype")) if isinstance(value, dict) else (value, None)
        cls = self._by_key.get(key)
        if cls is None:
            raise ValueError(f"{source}/{label}: unknown class {key!r}")
        if subtype is not None and subtype not in cls.subtypes:
            raise ValueError(f"{source}/{label}: {subtype!r} is not a sub-type of {key}")
        return LabelMapping(key, subtype)

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(c.key for c in self.classes)

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(self._sources)

    def get(self, key: str) -> DefectClass:
        return self._by_key[key]

    def by_id(self, class_id: int) -> DefectClass:
        return self._by_id[class_id]

    def source_labels(self, source: str) -> dict[str, LabelMapping]:
        return dict(self._sources[source])

    def map_label(self, source: str, label: str) -> LabelMapping:
        """Map a dataset label to the taxonomy; raises KeyError for unmapped labels."""
        try:
            return self._sources[source][label.strip().lower()]
        except KeyError:
            raise KeyError(
                f"label {label!r} of source {source!r} is not mapped in configs/taxonomy.yaml"
            ) from None

    def coco_categories(self) -> list[dict]:
        return [{"id": c.id, "name": c.key, "supercategory": "defect"} for c in self.classes]


@cache
def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or CONFIG_DIR / "taxonomy.yaml"
    with open(path, encoding="utf-8") as f:
        return Taxonomy(yaml.safe_load(f))
