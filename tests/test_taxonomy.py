import pytest
import yaml

from inspection.paths import CONFIG_DIR
from inspection.taxonomy import Taxonomy, load_taxonomy


def test_classes_match_plan():
    tax = load_taxonomy()
    assert tax.keys == (
        "scratch",
        "crack",
        "deformation",
        "pit_void",
        "missing_material",
        "contamination",
        "discoloration",
        "unknown",
        "paint_finish",
    )
    assert [c.id for c in tax.classes] == list(range(1, 10))
    assert tax.severity_levels == ("minor", "major", "critical")


@pytest.mark.parametrize(
    ("source", "label", "zd_class", "subtype"),
    [
        ("realiad", "OK", None, None),
        ("realiad", "AK", "pit_void", "pit"),
        ("realiad", "hs", "scratch", None),
        ("realiad", "QS", "missing_material", None),
        ("realiad", "YW", "contamination", "foreign_body"),
        ("paintdefect", "Dent", "deformation", "dent"),
        ("paintdefect", "fiber", "contamination", "fibre"),
        ("paintdefect", "sagging", "paint_finish", "sag"),
        ("mvtec_ad", "color", "discoloration", None),
        ("mvtec_ad2", "bad", "unknown", None),
    ],
)
def test_map_label(source, label, zd_class, subtype):
    m = load_taxonomy().map_label(source, label)
    assert (m.zd_class, m.subtype) == (zd_class, subtype)
    assert m.is_good == (zd_class is None)


def test_unmapped_label_raises():
    with pytest.raises(KeyError, match="not mapped"):
        load_taxonomy().map_label("realiad", "XX")


def test_every_realiad_code_is_mapped():
    assert set(load_taxonomy().source_labels("realiad")) == {
        "ok",
        "ak",
        "bx",
        "ch",
        "hs",
        "ps",
        "qs",
        "yw",
        "zw",
    }


def test_invalid_subtype_rejected():
    data = yaml.safe_load((CONFIG_DIR / "taxonomy.yaml").read_text())
    data["sources"]["realiad"]["AK"] = {"class": "pit_void", "subtype": "bogus"}
    with pytest.raises(ValueError, match="sub-type"):
        Taxonomy(data)


def test_coco_categories_use_class_ids():
    cats = load_taxonomy().coco_categories()
    assert cats[0] == {"id": 1, "name": "scratch", "supercategory": "defect"}
