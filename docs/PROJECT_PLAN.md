# ZeroDefect — Project Plan

**AI visual defect detection & quality intelligence for automobile polymer (plastic) parts**

Based on the project abstract *"AI-Based Visual Defect Detection System for Automobile Parts / Nut-Bolt Production Lines"*.
Scope for this build: **injection-moulded polymer automotive parts** (plugs, caps, clips, plastic nuts, trims, painted plastic panels), with metal nut/bolt support as an extension.

---

## 1. What we are building

A software-only system with three layers:

| Layer | What it does |
|---|---|
| **Inspection engine** (Python, CPU-friendly) | Takes camera frames (or recorded images), locates the part, aligns it, checks only the critical regions, finds and classifies defects, scores severity, fuses several views/frames, then gives PASS / FAIL with evidence. |
| **Quality platform** (FastAPI + PostgreSQL) | Stores a traceability record for every part: Part ID, machine, mould/cavity, operator, shift, resin batch, supplier, process parameters, defects, mm coordinates, confidence and images. Runs analytics, alerts, predictions, reports and the AI copilot. |
| **Operator/engineer UI** (React web app) | A modern dark "control-room" interface with live inspection, dashboards, traceability search, heatmaps, factory map, analytics, reports and a chat copilot. Views depend on the user's role. |

---

## 2. Real datasets

No single public dataset covers polymer car parts **and** production metadata, so we combine several real datasets, each chosen for a specific job.

### 2.1 Core datasets (use now)

| Dataset | Why we use it | Content | Labels | License / access |
|---|---|---|---|---|
| **Real-IAD** (CVPR 2024) | **Main dataset.** Real plastic/rubber industrial parts, including `plastic_nut`, `plastic_plug`, `end_cap`, `u_block` and `mounts`. Each part is shot from **5 camera angles**, which gives us real data for the multi-view / temporal-consistency stage. | 30 object classes, ~150K high-res images | Good/defect + pixel masks. 8 defect families: pit, deformation, abrasion, scratch, damage, missing parts, foreign object, contamination | CC BY-NC-SA 4.0. Gated: you request access, then download. We only download the polymer classes. |
| **MVTec AD 2** (2025) | Robustness to lighting. Its `wallplugs` class is a polymer part, and its test sets are captured under **changing lighting**. | 8 scenarios, 8K+ images | Pixel masks | CC BY-NC-SA 4.0 |
| **PaintDefect** (Roboflow Universe) | **Fine-grained, supervised defect types on painted plastic car parts** | Paint defects on moulded plastic car parts | Bounding boxes: scratch, dent, bump, dust, fibre | Check the license on the dataset page before use |
| **MVTec AD** (`screw`, `metal_nut`) | Nut-bolt extension from the abstract | 15 classes, 5.3K images | Pixel masks | CC BY-NC-SA 4.0 |

### 2.2 Optional / later

- **Tyre datasets (Mendeley):** "Digital images of defective and good condition tyres" (1,854 images) and TyreNet (1,698 images). Covers rubber, which is also a polymer.
- **Roboflow "molding defect" datasets:** more injection-moulding defect classes (short shot, flash, etc.). Quality varies, so we vet each one before use.
- **Zenodo PP injection-moulding dataset:** 14,620 images of polypropylene parts with segmentation masks, CAD, and the **injection process parameters** for every batch. It would be ideal for root-cause analysis, but the files only become public on **31 Dec 2027**. The pipeline is designed so it can be added then.

### 2.3 Unified defect taxonomy (polymer)

Every source label is mapped to one project taxonomy. Each type has sub-types and a severity scale.

| ZeroDefect class | Sub-types | Mapped from |
|---|---|---|
| Scratch | light / deep / edge / surface | Real-IAD `HS` (scratch), `CH` (abrasion); PaintDefect `scratch` |
| Crack / breakage | hairline / through / edge | Real-IAD `PS` (damage) |
| Dent / deformation / warpage | dent, bump, warp | Real-IAD `BX`; PaintDefect `dent`, `bump` |
| Pit / void / sink mark | pit, void, sink | Real-IAD `AK` |
| Missing material (short shot) | missing feature, missing hole | Real-IAD `QS` |
| Contamination / foreign material | black speck, dust, fibre, foreign body | Real-IAD `YW`, `ZW`; PaintDefect `dust`, `fibre` |
| Discoloration / burn mark | discoloration, burn, flow line | synthetic + future datasets |
| **Unknown** | — | anomaly model fires but no known class → sent to clustering |

### 2.4 Traceability data (simulated, and we say so openly)

Public datasets have **no machine, operator, shift, batch or supplier data**. A **production-line simulator** fills this gap. It assigns every real image to a virtual part record:

- 3 lines, 6 injection-moulding machines (`IMM-01…06`), 2-cavity moulds, 9 operators, 3 shifts, 4 resin suppliers and their batches
- Process parameters for each shot: melt temperature, injection pressure, holding time, cooling time, mould temperature, cycle time
- **Planted faults** that the analytics must discover on their own. For example: a mould-temperature drift on IMM-03 that raises short shots, or a contaminated resin batch from Supplier C.

This gives the dashboards, root-cause analysis, predictions and copilot realistic data to work on. Because the planted faults are known, we can **measure** whether root-cause analysis finds the true cause. The report must state clearly that this metadata is simulated. The images and defects are real.

---

## 3. Inspection pipeline (maps to abstract §4)

```
Camera / video / image folder
   │
   ▼
[1] Part detection ─► [2] Alignment to golden template ─► [3] Critical-region (ROI) masks
                                                               │
                        ┌──────────────────────────────────────┤
                        ▼                                      ▼
          [4a] Anomaly model (trained on       [4b] Supervised defect detector
               GOOD parts only)                     (type + box/mask)
               PatchCore / EfficientAD              YOLO-seg or RF-DETR
               → anomaly heatmap                    → scratch, dent, …
                        └──────────────┬───────────────────────┘
                                       ▼
                  [5] Fusion + context-aware severity
                      (size in mm², length in mm, inside which ROI,
                       distance to hole/edge, confidence)
                                       ▼
                  [6] Multi-view / multi-frame consistency
                      (Real-IAD 5 views; ByteTrack for video)
                                       ▼
                  PASS / FAIL + evidence + traceability record
```

| Abstract principle | How it is implemented |
|---|---|
| 4.1 Multi-stage | The staged pipeline above. Stages are pluggable Python classes with a per-part-type config (YAML). |
| 4.2 Fine-grained classification | Detector classes + sub-type rules (e.g. scratch depth from local contrast/width, edge vs surface from ROI) + severity (minor / major / critical). |
| 4.3 Synthetic defects | Procedural scratches, cracks and pits (OpenCV + Perlin noise, DRAEM-style); CutPaste of real defects onto good parts; optional diffusion inpainting for realism. |
| 4.4 Augmentation | Albumentations: rotation, flip, scale, brightness/contrast, blur, motion blur, shadow, specular glare, noise, dust. |
| 4.5 Temporal consistency | Confirm a defect only if it is seen in ≥ k of N views/frames, using a weighted vote on aligned coordinates. |
| 4.6 Context-aware | Critical ROIs (sealing faces, snap-fits, threads, holes) are drawn in the UI. The same scratch can be *acceptable* on a non-functional face and *critical* on a sealing face. |
| §5 Traceability | Unique Part ID at detection. Pixel→mm calibration from a checkerboard or a known part dimension. Full record saved to the database. |

**Why two models:** the anomaly model needs only good parts and catches **unknown** defects. The supervised detector gives **exact defect types**. Combining them solves the "few defect samples" problem. It also feeds the *Automatic Defect Clustering* feature: anomalies with no known class are embedded and grouped with HDBSCAN. An engineer then labels them in the UI, and those labels become training data for the next model version (active learning).

**Deployment:** models are exported to **ONNX / OpenVINO**, so inference runs on an ordinary CPU (target < 150 ms per view). A GPU is only needed for training.

---

## 4. Features (abstract §6), and how each is built

| Feature | Implementation |
|---|---|
| AI visual defect detection | Pipeline above; live feed over WebSocket |
| Defect classification | Detector + taxonomy + sub-type/severity rules |
| Machine-wise analysis | Defect rate, Pareto and control chart per machine/mould/cavity |
| Operator & shift performance | Normalised defect rates, with statistical significance tests (χ²) so the system does not unfairly blame an operator |
| Raw material & supplier quality | Defect rate by resin batch/supplier, incoming-lot ranking |
| Defect heatmap | Aligned defect coordinates accumulated on the part template (2D KDE overlay) |
| Production quality dashboard | Throughput, yield, first-pass yield, PPM, Pareto, trends — live |
| Root cause analysis | Correlation of defects with machine/batch/process parameters + polymer domain rules (e.g. short shot ↔ low pressure/melt temperature; flash ↔ high pressure/worn mould; sink ↔ short cooling) + SHAP on a gradient-boosting model |
| Real-time alerts | Rule engine (rate > threshold, Western Electric SPC rules, repeated defect at the same location), shown in the UI + optional email/webhook |
| Trend analysis | Daily/weekly/monthly aggregation, moving averages, change-point detection |
| Automated reports | PDF (WeasyPrint) + Excel (openpyxl): shift report, part certificate, supplier report |
| AI quality score | Score of 0–100 per line/machine from weighted defect rate, severity, trend and stability |
| Quality prediction | Next-shift/day defect-rate forecast (gradient boosting on lag features + process parameters; Prophet/ETS baseline) |
| AI recommendation engine | Maps root-cause findings to actions (recalibrate, adjust melt temperature, clean mould vents, quarantine batch), with expected impact |
| Automatic defect clustering | Embeddings of unknown anomalies → HDBSCAN → review-and-label screen |
| Live factory map | Floor-plan SVG of lines/machines with green/amber/red health status |
| Defect cost analysis | Configurable scrap/rework/downtime cost per defect type → cost Pareto |
| AI factory copilot | LLM with tool use over **read-only** SQL views and analytics functions ("Which machine had the most short shots last night shift, and why?"), with charts in its answers |

---

## 5. UI plan ("good UI")

**Stack:** React + TypeScript + Vite, Tailwind CSS + shadcn/ui, Apache ECharts, TanStack Query, Konva (canvas drawing for image overlays and the ROI editor). Dark industrial theme with a light mode option. Keyboard-friendly, and readable at a distance on a line-side monitor.

| # | Screen | Main user |
|---|---|---|
| 1 | **Live Inspection**: camera feed with defect boxes and heatmap overlay, large PASS/FAIL badge, Part ID, current machine/operator/shift, ribbon of recent parts, FPS/latency | Operator |
| 2 | **Part Traceability**: search by Part ID/time/batch; all views, defects with type, severity, mm location and confidence; full production context; export certificate | Engineer |
| 3 | **Production Dashboard**: KPI cards, yield trend, defect Pareto, live counters | Supervisor |
| 4 | **Analytics**: Machine / Operator & Shift / Supplier & Batch tabs | Engineer |
| 5 | **Defect Heatmap** on the part template, with filters | Engineer |
| 6 | **Factory Map** | Supervisor |
| 7 | **Alerts Center** + rule editor | Supervisor |
| 8 | **Root Cause & Recommendations** | Engineer |
| 9 | **Predictions & Quality Score** | Manager |
| 10 | **Unknown Defect Clusters**: review, label, send to training | Engineer |
| 11 | **Cost Analysis** | Manager |
| 12 | **Reports**: generate and download PDF/Excel | All |
| 13 | **AI Copilot**: chat panel, available on every page | All |
| 14 | **Setup**: part types, ROI editor, severity rules, thresholds, calibration, cameras, models, users/roles | Admin |

Roles: **Operator, Supervisor, Quality Engineer, Admin** (JWT auth).

---

## 6. Tech stack

- **ML:** Python 3.11, PyTorch, anomalib 2.x (PatchCore, EfficientAD), Ultralytics YOLO-seg *(AGPL-3.0; fine for academic use, needs a license for commercial use)* or RF-DETR *(Apache-2.0)*, OpenCV, Albumentations, scikit-learn (HDBSCAN, gradient boosting), ONNX Runtime / OpenVINO
- **Backend:** FastAPI, SQLAlchemy 2, PostgreSQL (SQLite for quick local demo), WebSockets, background workers, WeasyPrint, openpyxl
- **Frontend:** React, TypeScript, Vite, Tailwind, shadcn/ui, ECharts, Konva
- **Copilot:** Claude API (tool use + read-only SQL), with an offline mode that answers from pre-built analytics
- **DevOps:** Docker Compose (db + api + worker + web), pytest, Playwright UI tests, GitHub Actions CI

### Repository layout
```
ZeroDefect/
├── ml/              datasets (download + convert), synth defects, augmentation, training, evaluation, export
├── inspection/      pipeline stages, calibration, fusion, severity rules, part-type configs
├── simulator/       production-line + traceability metadata generator, virtual camera
├── backend/         FastAPI app, DB models, services, analytics, alerts, reports, copilot
├── frontend/        React UI
├── deploy/          docker-compose, env templates
├── notebooks/       Colab/Kaggle training notebooks (GPU)
└── docs/            this plan, dataset cards, evaluation report, user guide
```

---

## 7. Evaluation (what goes in the project report)

- **Detection:** image AUROC, pixel AUROC, AUPRO (anomaly model); mAP@50 / mAP@50-95 and per-class precision/recall/F1 (detector)
- **Line metrics:** false-reject rate (good parts rejected) and **escape rate** (defective parts passed) at the chosen operating threshold
- **Speed:** ms per part on CPU (OpenVINO) vs GPU
- **Ablation study**, which proves the abstract's claims with numbers:
  1. single model vs multi-stage + ROI
  2. no synthetic data vs + synthetic defects
  3. no augmentation vs + augmentation (tested on MVTec AD 2 changing-light sets)
  4. single view vs multi-view fusion (Real-IAD 5 views)
- **Root-cause accuracy:** does root-cause analysis recover the planted simulator faults? (top-1 / top-3 hit rate)
- **Forecast error:** MAE/MAPE of the next-shift defect-rate prediction

---

## 8. Milestones & time estimate

The estimate covers **my build time**: writing, testing and debugging the code in these sessions. **Calendar time** also includes your parts: dataset access approvals, GPU training runs, and reviewing each milestone.

| # | Milestone | Deliverable | My build time |
|---|---|---|---|
| M0 | Foundation | Repo scaffold, dataset download/convert scripts, unified taxonomy, production simulator, virtual camera | 3–5 h |
| M1 | Inspection engine | Detection → alignment → ROI → PatchCore/EfficientAD → severity → multi-view fusion → PASS/FAIL, with calibration | 6–9 h |
| M2 | Data & models | Synthetic defect generator, augmentation, detector training notebooks, evaluation + ablation harness, ONNX/OpenVINO export | 5–8 h *(+ 4–10 h of GPU time you run on Colab/Kaggle)* |
| M3 | Backend | DB schema, traceability API, ingestion service, WebSocket live stream, auth/roles, alerts engine | 5–7 h |
| M4 | UI | Design system + all 14 screens, ROI editor, live inspection view | 9–13 h |
| M5 | Intelligence | Root cause, recommendations, quality score, prediction, clustering + labelling loop, cost analysis, PDF/Excel reports | 5–8 h |
| M6 | AI Copilot | Tool-using LLM over read-only data, chat UI with inline charts | 2–4 h |
| M7 | Hardening & delivery | Tests, CI, Docker one-command start, user guide, evaluation report, demo script | 3–5 h |
| | **Total** | | **≈ 38–59 hours of build time** |

**Realistic calendar:** about **3–5 weeks**, working in sessions of a few hours with your review between milestones.
- **First demo in week 1:** live inspection of Real-IAD `plastic_nut`, with a PatchCore heatmap, PASS/FAIL, and a basic dashboard.
- **Everything working end-to-end by week 3.**
- **Weeks 4–5:** polish, ablation numbers, and the report.

For comparison, one developer building this without AI assistance would typically need **4–6 months**.

---

## 9. What I need from you before M0

Step-by-step instructions are in **[docs/SETUP.md](SETUP.md)**. In short:

1. **Network access:** set the cloud environment's network access to **Full**, or **Custom** with the dataset domains listed in SETUP.md.
2. **Dataset access:** request Real-IAD access on Hugging Face (approval is automatic), then store `HF_TOKEN`. Also store `ROBOFLOW_API_KEY`, and fill in the MVTec AD 2 download form.
3. **GPU for training:** a free Colab or Kaggle GPU. This environment has 4 CPUs, 15 GB RAM and no GPU, which is fine for PatchCore and CPU inference but not for detector training.
4. **Use: academic (decided).** The non-commercial datasets (CC BY-NC-SA) and Ultralytics YOLO (AGPL) are used as planned.
5. **Copilot API key (optional):** store it as `ZERODEFECT_ANTHROPIC_API_KEY`, not `ANTHROPIC_API_KEY`, which Claude Code itself reads. Without it, the copilot runs in offline mode.

---

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Real-IAD is large (high-res, 5 views) | Download only the polymer classes; resize to 1024 px during conversion |
| Datasets use different label formats | One converter per source into a single internal format (COCO + masks + taxonomy mapping) |
| Traceability metadata is not real | Transparent simulator with planted faults; the report states this clearly; the schema accepts real MES data unchanged |
| No physical camera line | Virtual camera replays the datasets at line speed; a webcam/RTSP/video file also works for a live demo |
| CPU-only inference speed | OpenVINO export, ROI cropping, smaller backbones; GPU optional |
| License limits | Academic use only unless replaced with self-collected data |

---

### Dataset references
- Real-IAD paper: https://arxiv.org/abs/2403.12580 — download: https://huggingface.co/datasets/Real-IAD/Real-IAD — anomalib datamodule: https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/realiad.html
- MVTec AD 2: https://arxiv.org/abs/2503.21622 — downloads: https://www.mvtec.com/company/research/datasets/mvtec-ad-2/downloads
- PaintDefect (Roboflow Universe): https://universe.roboflow.com/ai-klghd/paintdefect-8h4s4
- Tyre datasets: https://data.mendeley.com/datasets/bn7ch8tvyp , https://data.mendeley.com/datasets/32b5vfj6tc
- Zenodo PP injection-moulding dataset (public 31 Dec 2027): https://zenodo.org/records/20322729
