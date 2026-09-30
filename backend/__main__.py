"""Start the ZeroDefect web app.

uv run python -m backend                 # http://127.0.0.1:8000
uv run python -m backend --speed 10      # simulated time runs 10x faster than real time
"""

import argparse
import logging
import webbrowser

import uvicorn

from backend.app import Plant, create_app
from inspection.paths import data_root


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m backend", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--images", default="demo", help="dataset sources for part images (default: demo)")
    p.add_argument("--speed", type=float, default=5.0, help="simulated seconds per real second (default: 5)")
    p.add_argument("--history-days", type=int, default=7, help="days of production before now (default: 7)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    images = args.images.split(",")
    if images == ["demo"] and not any((data_root() / "processed/demo").glob("*/manifest.jsonl")):
        from ml.datasets import demo

        print("Creating the demo parts dataset (first run only)...")
        demo.generate()
    print(f"Simulating {args.history_days} days of production...")
    app = create_app(Plant(images, args.history_days, args.speed))
    url = f"http://127.0.0.1:{args.port}"
    print(f"ZeroDefect is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
