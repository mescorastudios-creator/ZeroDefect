"""Defect detector command line.

python -m ml.detector build --sources realiad paintdefect --name polymer
python -m ml.detector train --data polymer --model yolo26s.pt --epochs 100 --imgsz 640 --device 0
python -m ml.detector eval --weights outputs/detector/polymer/weights/best.pt --data polymer
python -m ml.detector export --weights outputs/detector/polymer/weights/best.pt --format onnx

--data takes a dataset name (data/yolo/<name>/data.yaml) or a path to a data.yaml.
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

from ml.detector import dataset


def _data_yaml(value: str) -> Path:
    path = Path(value)
    if path.suffix in (".yaml", ".yml"):
        return path.resolve()
    return dataset.yolo_dir(value).resolve() / "data.yaml"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m ml.detector",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="write a YOLO dataset from converted sources")
    b.add_argument("--sources", nargs="+", required=True, help="converted sources, e.g. realiad paintdefect")
    b.add_argument("--categories", nargs="+", help="only these part types (default: all converted)")
    b.add_argument("--name", required=True, help="dataset name: data/yolo/<name>")
    b.add_argument("--task", default="detect", choices=dataset.TASKS)
    b.add_argument(
        "--background-ratio",
        type=float,
        default=dataset.BACKGROUND_RATIO,
        help="defect-free images per defect image in train/val",
    )

    t = sub.add_parser("train", help="fine-tune a YOLO model")
    t.add_argument("--data", required=True)
    t.add_argument("--model", default="yolo26s.pt", help="pretrained weights or model yaml")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--device", help="e.g. 0 for the first GPU, cpu")
    t.add_argument("--name", help="run name: outputs/detector/<name> (default: the dataset name)")
    t.add_argument("--workers", type=int, help="data loader workers")
    t.add_argument("--fraction", type=float, help="train on this share of the train split (quick tests)")
    t.add_argument("--cache", choices=["ram", "disk"], help="cache decoded images")
    t.add_argument("--time", type=float, help="stop after this many hours (overrides --epochs)")
    t.add_argument(
        "--set",
        nargs="+",
        default=[],
        metavar="KEY=VALUE",
        help="any other Ultralytics train argument, e.g. close_mosaic=2 cos_lr=true",
    )

    for name in ("eval", "export"):
        e = sub.add_parser(name, help=f"{name} trained weights")
        e.add_argument("--weights", required=True, type=Path)
        e.add_argument("--imgsz", type=int, default=640)
        if name == "eval":
            e.add_argument("--data", required=True)
            e.add_argument("--split", default="test", choices=dataset.SPLITS)
            e.add_argument("--conf", type=float, default=0.25, help="pass/fail threshold for line metrics")
            e.add_argument("--device")
            e.add_argument("--batch", type=int, default=16)
        else:
            e.add_argument("--format", nargs="+", default=["onnx"], help="onnx, openvino, ...")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        _run(args)
    except (FileNotFoundError, ImportError, ValueError) as e:
        p.exit(1, f"error: {e}\n")


def _run(args) -> None:
    if args.cmd == "build":
        dataset.build(
            args.name, args.sources, args.categories, task=args.task, background_ratio=args.background_ratio
        )
        return

    from ml.detector import train as det

    if args.cmd == "train":
        data = _data_yaml(args.data)
        if not data.exists():
            raise FileNotFoundError(f"{data} not found; run python -m ml.detector build first")
        extra = {
            k: getattr(args, k)
            for k in ("workers", "fraction", "cache", "time")
            if getattr(args, k) is not None
        }
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
            extra[key] = yaml.safe_load(value)
        best = det.train(
            data,
            args.model,
            args.epochs,
            args.imgsz,
            args.batch,
            args.device,
            args.name or data.parent.name,
            **extra,
        )
        print(f"best weights: {best}")
    elif args.cmd == "eval":
        r = det.evaluate(
            args.weights, _data_yaml(args.data), args.split, args.conf, args.imgsz, args.device, args.batch
        )
        print(json.dumps({"overall": r["overall"], "line": r["line"]}, indent=2))
    elif args.cmd == "export":
        for path in det.export(args.weights, args.format, args.imgsz):
            print(path)


if __name__ == "__main__":
    main()
