# ZeroDefect — Datasets

How each dataset is downloaded, converted to one internal format, and mapped to the defect taxonomy.

All commands run from the repository root. Data lives in `data/` (git-ignored); set `ZERODEFECT_DATA_ROOT` to put it elsewhere.

```bash
uv run python -m ml.datasets download <source> --convert   # download + convert
uv run python -m ml.datasets convert <source>              # re-convert what is on disk
uv run python -m ml.datasets demo                          # procedural demo set, no download
uv run python -m ml.datasets stats                         # counts per source/category/split/class
```

## Status

| Source | Classes | Status |
|---|---|---|
| Real-IAD | `plastic_nut`, `plastic_plug`, `end_cap`, `u_block`, `mounts` | Code ready; fixture-tested. **Download blocked:** the `HF_TOKEN` in the environment is rejected by Hugging Face. |
| MVTec AD | `screw`, `metal_nut` | Downloaded and converted. Counts match the published dataset (screw 320 train / 160 test, metal_nut 220 / 115). |
| MVTec AD 2 | `wallplugs` | Code ready; fixture-tested. Not downloaded yet (needed from M2). |
| PaintDefect | `painted_panel` | Code ready; fixture-tested. Needs `ROBOFLOW_API_KEY` (from M2). |
| Demo (procedural) | the 5 Real-IAD polymer part types | Generated locally. Not training data. |

## Sources

### Real-IAD (main dataset)
- **What:** real plastic/rubber industrial parts, each shot by 5 cameras (`C1`–`C5`). Defect codes: `AK` pit, `BX` deformation, `CH` abrasion, `HS` scratch, `PS` damage, `QS` missing parts, `YW` foreign object, `ZW` contamination.
- **Access:** gated on Hugging Face (`Real-IAD/Real-IAD`). Accept the terms, then set `HF_TOKEN`. License CC BY-NC-SA 4.0, research use.
- **Download:** `download realiad [--resolution 256|512|1024|raw] [--classes ...]`. The dataset ships pre-resized copies, so nothing is resized during conversion. Default 512 px (about 2.3 GB for the 5 polymer classes). Use 1024 px for detector training on a GPU (about 8.8 GB). Zips are deleted after extraction unless `--keep-archives`.
- **Split:** the official `realiad_jsons/realiad_jsons/<class>.json` (multi-view setting). If it is missing, the converter falls back to a deterministic split: 80/20 train/test for good samples, all defective samples in test.
- **Layout:** `realiad_<res>/<class>/OK/S0001/...jpg`, `NG/<code>/S0001/...jpg` with a same-stem `.png` mask.

### MVTec AD (nut/bolt extension)
- **Download:** `download mvtec_ad`. Streams the official 5.3 GB archive and extracts only `screw/` and `metal_nut/`. It stops as soon as both have been read (about 3.7 GB transferred), and resumes automatically after a dropped connection.
- **Layout:** `<class>/train/good`, `<class>/test/<defect>`, `<class>/ground_truth/<defect>/<stem>_mask.png`. License CC BY-NC-SA 4.0.

### MVTec AD 2 (lighting robustness)
- **Download:** `download mvtec_ad2`. By default it streams the full 32.7 GB archive (the link anomalib uses) and keeps only `wallplugs/`. The per-class link from the MVTec download form is much smaller: `download mvtec_ad2 --url <link>`.
- **Converted:** `train/good` → train, `validation/good` → val, `test_public/{good,bad}` → test.
- **Important for the M2 ablation:** the changing-lighting test sets (`test_private`, `test_private_mixed`) ship **without ground truth**. They are scored on the MVTec benchmark server (https://benchmark.mvtec.com/). Local lighting-robustness numbers therefore need either that server or lighting perturbations applied to `test_public`.

### PaintDefect (supervised defect types, painted plastic)
- **Download:** `download paintdefect [--version N]`. Uses the Roboflow REST API with `ROBOFLOW_API_KEY` and fetches the COCO export (latest version by default). Check the license on the dataset page first.
- **Converted:** one category, `painted_panel`. Boxes are kept as they are. The image-level class is the image's most frequent defect class, and all boxes go into `coco.json`.
- Labels missing from the taxonomy make conversion fail with the list of labels, so that no label is silently dropped.

### Demo (procedural)
`demo` draws the 5 polymer part types with one defect of each class and a pixel mask, 5 views per part. Train split: 40 good parts (anomaly model) and 6 labelled parts per defect class (defect-type classifier); test split: 20 good + 4 per class, held out. It exists so the simulator, virtual camera, tests and CI work without downloads. **Never report metrics on it.**

## Unified format

Each converted `<source>/<category>` gets `data/processed/<source>/<category>/`:

- **`manifest.jsonl`**: one line per image.

  | Field | Meaning |
  |---|---|
  | `image_id` | unique id, `<source>/<category>/...` |
  | `source`, `category` | dataset and part type (e.g. `realiad`, `plastic_nut`) |
  | `split` | `train` / `val` / `test` |
  | `image_path`, `mask_path` | relative to the data root |
  | `label` | `good` / `defect` |
  | `zd_class`, `zd_subtype` | taxonomy class and sub-type (null when good or unknown) |
  | `source_label` | original label, kept for traceability (e.g. `AK`) |
  | `sample_id`, `view` | physical part and camera view. The views of one part share a `sample_id`. |
  | `width`, `height`, `extra` | image size, source-specific extras |

- **`coco.json`**: COCO instances for detector training. Category ids are the taxonomy class ids (`scratch`=1 … `unknown`=8). Instances come from source boxes (PaintDefect) or from mask regions. Regions closer than 5 px count as one defect, and specks under 4 px are dropped. Image records carry `split`, `label`, `sample_id` and `view`.

`ml.datasets.common.load_manifests()` loads every manifest into one DataFrame.

## Taxonomy

[`configs/taxonomy.yaml`](../configs/taxonomy.yaml) is the single source of truth: classes, sub-types, default severity, colours, and every source-label mapping. Summary:

| Class (id) | Real-IAD | PaintDefect | MVTec AD | MVTec AD 2 |
|---|---|---|---|---|
| scratch (1) | `HS`, `CH` | scratch | scratch_head, scratch_neck, scratch | |
| crack (2) | `PS` | | thread_side, thread_top | |
| deformation (3) | `BX` | dent, bump | manipulated_front, bent | |
| pit_void (4) | `AK` | | | |
| missing_material (5) | `QS` | | | |
| contamination (6) | `YW`, `ZW` | dust, fibre | | |
| discoloration (7) | | | color | |
| unknown (8) | | | flip (orientation fault) | bad (untyped) |

Sub-types are only set when the source label names one (`AK`→pit, dent/bump, dust/fibre, `YW`→foreign_body). Geometric sub-types, such as scratch depth or edge vs surface, are derived by the inspection engine in M1.
