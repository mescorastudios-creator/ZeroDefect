"""Simulator command line.

python -m simulator run --images demo            # 14-day history bound to demo images
python -m simulator run --images realiad --days 28
python -m simulator run --images none            # metadata only
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from inspection.paths import data_root
from simulator.config import load_config
from simulator.images import ImagePool
from simulator.production import ProductionSimulator, fault_effects, save_run


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m simulator", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate a production history")
    r.add_argument("--config", type=Path, help="plant config (default: configs/plant.yaml)")
    r.add_argument("--days", type=int, help="override the configured number of days")
    r.add_argument("--seed", type=int, help="override the configured seed")
    r.add_argument(
        "--images",
        default="demo",
        help="comma-separated dataset sources to bind images from, or 'none' (default: demo)",
    )
    r.add_argument("--split", default="test", help="dataset split the images come from (default: test)")
    r.add_argument("--out", type=Path, help="output folder (default: data/sim/<images>)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.days:
        cfg.days = args.days
    if args.seed is not None:
        cfg.seed = args.seed
    pool = None if args.images == "none" else ImagePool.from_sources(args.images.split(","), args.split)
    sim = ProductionSimulator(cfg, pool)
    parts = sim.run()
    out = args.out or data_root() / "sim" / args.images.replace(",", "+")
    save_run(out, sim, parts)

    defects = parts[parts.true_label == "defect"]
    print(
        f"\n{len(parts):,} parts over {cfg.days} days, defect rate {len(defects) / len(parts):.2%} -> {out}\n"
    )
    by_machine = pd.crosstab(parts.machine_id, parts.true_class.fillna("good"), normalize="index") * 100
    print("Defect rate by machine and class (%):")
    print(by_machine.drop(columns="good").round(2).to_string(), "\n")
    print("Planted faults (target-defect rate inside scope vs. control):")
    print(fault_effects(parts, cfg, sim.start).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
