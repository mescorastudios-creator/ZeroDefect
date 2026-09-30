# ZeroDefect

AI visual defect detection and quality intelligence for automobile polymer (plastic) parts.

A software-only inspection system: it detects, classifies and traces defects on moulded plastic parts using existing line cameras. On top of that it offers dashboards, root-cause analysis, predictions and an AI copilot.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full plan: datasets, architecture, UI, milestones and time estimate.

## Run the web app

1. Install [uv](https://docs.astral.sh/uv/) (it installs Python for you):
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Get the code: `git clone` the repository, or download it from GitHub as a ZIP (Code → Download ZIP) and unzip it.
3. In the project folder run:
   ```bash
   uv run python -m backend
   ```
   The browser opens at http://127.0.0.1:8000. The first run needs internet: it downloads Python and the packages, then creates the demo part images (about a minute). After that it also works offline.

Screens: **Live inspection** (parts streaming from the line, PASS/FAIL, 5 camera views, alerts), **Dashboard**, **Factory map** (machine health, click a machine for its history) and **Traceability** (search by part ID, resin lot or operator). The line runs 5× faster than real time; change it with `--speed 10`. Stop with Ctrl+C.

Production data is simulated, and PASS/FAIL comes from the dataset labels until the AI model is connected (milestone M1).

## Quick start (development)

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                         # install
uv run python -m ml.datasets demo               # small procedural dataset, no download
uv run python -m simulator run --images demo    # 14 days of simulated production -> data/sim/demo/
uv run python -m simulator.camera --source sim --images demo --speed 60 --limit 20
uv run pytest                                   # tests
```

Real datasets (access and keys: [docs/SETUP.md](docs/SETUP.md)):

```bash
uv run python -m ml.datasets download realiad --convert    # needs HF_TOKEN
uv run python -m ml.datasets download mvtec_ad --convert
uv run python -m simulator run --images realiad
```

## Layout

| Folder | Content | Milestone |
|---|---|---|
| `configs/` | Defect taxonomy and plant configuration (single sources of truth) | M0 |
| `inspection/` | Taxonomy loader; inspection pipeline stages | M0, M1 |
| `ml/datasets/` | Dataset download + conversion into one format, procedural demo set | M0 |
| `simulator/` | Production-line simulator with planted faults, virtual camera | M0 |
| `backend/` | Web app: FastAPI backend + UI (first version; M3/M4 extend it) | M3, M4 |
| `frontend/`, `deploy/`, `notebooks/` | React UI, deployment, GPU training | M2–M7 |
| `docs/` | [Plan](docs/PROJECT_PLAN.md), [setup](docs/SETUP.md), [datasets](docs/DATASETS.md), [simulator](docs/SIMULATOR.md) | |

Environment variables: `HF_TOKEN` (Real-IAD), `ROBOFLOW_API_KEY` (PaintDefect), `ZERODEFECT_ANTHROPIC_API_KEY` (copilot), `ZERODEFECT_DATA_ROOT` (data folder, default `./data`).
