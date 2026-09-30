# ZeroDefect

AI visual defect detection and quality intelligence for automobile polymer (plastic) parts.

A software-only inspection system: it detects, classifies and traces defects on moulded plastic parts using existing line cameras. On top of that it offers dashboards, root-cause analysis, predictions and an AI copilot.

See [docs/PROJECT_PLAN.md](docs/PROJECT_PLAN.md) for the full plan: datasets, architecture, UI, milestones and time estimate.

## Quick start

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
| `backend/`, `frontend/`, `deploy/`, `notebooks/` | Quality platform, UI, deployment, GPU training | M2–M7 |
| `docs/` | [Plan](docs/PROJECT_PLAN.md), [setup](docs/SETUP.md), [datasets](docs/DATASETS.md), [simulator](docs/SIMULATOR.md) | |

Environment variables: `HF_TOKEN` (Real-IAD), `ROBOFLOW_API_KEY` (PaintDefect), `ZERODEFECT_ANTHROPIC_API_KEY` (copilot), `ZERODEFECT_DATA_ROOT` (data folder, default `./data`).
