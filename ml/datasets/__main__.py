"""Dataset command line.

python -m ml.datasets download realiad --resolution 512
python -m ml.datasets convert realiad
python -m ml.datasets demo
python -m ml.datasets stats
"""

import argparse
import logging

from ml.datasets import demo, mvtec_ad, mvtec_ad2, paintdefect, realiad
from ml.datasets.common import load_manifests

MODULES = {"realiad": realiad, "mvtec_ad": mvtec_ad, "mvtec_ad2": mvtec_ad2, "paintdefect": paintdefect}
DEFAULT_CLASSES = {
    "realiad": realiad.POLYMER_CLASSES,
    "mvtec_ad": mvtec_ad.CLASSES,
    "mvtec_ad2": mvtec_ad2.CLASSES,
}


def _source_kwargs(args) -> dict:
    kw = {}
    if args.source in DEFAULT_CLASSES:
        kw["classes"] = args.classes or DEFAULT_CLASSES[args.source]
    if args.source == "realiad":
        kw["resolution"] = args.resolution
    if args.source == "paintdefect" and args.version:
        kw["version"] = args.version
    return kw


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m ml.datasets",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("download", "convert"):
        sp = sub.add_parser(name, help=f"{name} a source dataset")
        sp.add_argument("source", choices=MODULES)
        sp.add_argument("--classes", nargs="+", help="dataset classes (default: the ones the plan uses)")
        sp.add_argument(
            "--resolution",
            default=realiad.DEFAULT_RESOLUTION,
            choices=realiad.RESOLUTIONS,
            help="Real-IAD image size",
        )
        sp.add_argument("--version", type=int, help="PaintDefect Roboflow version (default: latest)")
        if name == "download":
            sp.add_argument(
                "--url", help="MVTec archive URL override (e.g. the per-class link from the form)"
            )
            sp.add_argument("--convert", action="store_true", help="convert right after downloading")
    sd = sub.add_parser("demo", help="generate the procedural demo dataset")
    sd.add_argument("--size", type=int, default=256)
    sd.add_argument("--seed", type=int, default=0)
    sub.add_parser("stats", help="count converted samples per source/category/split/class")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        _run(p, args)
    except (PermissionError, FileNotFoundError) as e:
        p.exit(1, f"error: {e}\n")


def _run(p: argparse.ArgumentParser, args) -> None:
    if args.cmd == "download":
        kw = _source_kwargs(args)
        if args.url:
            if not args.source.startswith("mvtec"):
                p.error("--url only applies to mvtec_ad and mvtec_ad2")
            kw["url"] = args.url
        MODULES[args.source].download(**kw)
        if args.convert:
            kw.pop("url", None)
            MODULES[args.source].convert(**kw)
    elif args.cmd == "convert":
        MODULES[args.source].convert(**_source_kwargs(args))
    elif args.cmd == "demo":
        demo.generate(size=args.size, seed=args.seed)
    elif args.cmd == "stats":
        df = load_manifests()
        if df.empty:
            print("no converted datasets under data/processed")
            return
        df["zd_class"] = df["zd_class"].fillna("good")
        table = df.groupby(["source", "category", "split", "zd_class"]).size().unstack(fill_value=0)
        print(table.to_string())


if __name__ == "__main__":
    main()
