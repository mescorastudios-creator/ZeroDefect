"""Start the ZeroDefect web app.

uv run python -m backend                 # http://127.0.0.1:8000
uv run python -m backend --interval 3    # inspect a part every 3 seconds
"""

import argparse
import logging
import socket
import threading
import time
import webbrowser

import uvicorn

from backend.app import Plant, create_app
from inspection.paths import data_root
from ml.datasets import demo
from ml.datasets.common import load_manifests


def _open_when_ready(url: str, port: int) -> None:
    """Open the browser only once the server accepts connections."""
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            webbrowser.open(url)
            return
        except OSError:
            time.sleep(0.2)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m backend", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--images", default="demo", help="dataset sources for part images (default: demo)")
    p.add_argument(
        "--interval", type=float, default=1.5, help="seconds between inspected parts (default: 1.5)"
    )
    p.add_argument("--history-days", type=int, default=7, help="days of placeholder history (default: 7)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    images = args.images.split(",")
    manifest = load_manifests(["demo"])
    if images == ["demo"] and not ((manifest.split == "train") & (manifest.label == "defect")).any():
        print("Creating the demo parts dataset (first run only)...")
        demo.generate()
        for stale in (data_root() / "models").glob("demo_*.pt"):
            stale.unlink()
    categories = load_manifests(images[:1]).category.unique()
    if not all((data_root() / "models" / f"{images[0]}_{c}.pt").exists() for c in categories):
        print("Training the defect-finding model (first run only, about 1-2 minutes)...")
    print(f"Loading the model and {args.history_days} days of placeholder production data...")
    app = create_app(Plant(images, args.history_days, args.interval))
    url = f"http://127.0.0.1:{args.port}"
    print(f"ZeroDefect is running at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Thread(target=_open_when_ready, args=(url, args.port), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", timeout_graceful_shutdown=1)


if __name__ == "__main__":
    main()
