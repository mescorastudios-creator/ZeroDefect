# ZeroDefect — Defect detector (M2)

Pipeline stage 4b: a supervised YOLO detector that names the defect type and draws its box. The anomaly model (PatchCore, stage 4a) learns from good parts only and also catches unknown defects; the detector gives exact types. Stage 5 fuses the two.

Code: `ml/detector/`. Model: **Ultralytics YOLO26** (NMS-free; AGPL-3.0, fine for this academic project). Install the training packages with `uv sync --group train`; the web app and CI do not need them.

```bash
uv run python -m ml.detector build --sources realiad paintdefect --name polymer
uv run python -m ml.detector train --data polymer --model yolo26s.pt --imgsz 1024 --device 0
uv run python -m ml.detector eval --weights outputs/detector/polymer/weights/best.pt --data polymer
uv run python -m ml.detector export --weights outputs/detector/polymer/weights/best.pt
```

GPU training: [`notebooks/train_detector.ipynb`](../notebooks/train_detector.ipynb) runs all four steps on Colab or Kaggle ([open in Colab](https://colab.research.google.com/github/mescorastudios-creator/ZeroDefect/blob/claude/magical-feynman-bx2o1g/notebooks/train_detector.ipynb)). It needs the `HF_TOKEN` and `ROBOFLOW_API_KEY` secrets (docs/SETUP.md, steps 1, 2 and 6).

## 1. Dataset (`build`)

Reads the unified format (`data/processed/*/*/manifest.jsonl` + `coco.json`) and writes `data/yolo/<name>/`: `images/` (hard links, so no extra disk space), `labels/`, `data.yaml` and `index.csv` (one row per image: split, physical part, view, label, box count).

- **Classes:** YOLO class = taxonomy id − 1, all 9 taxonomy classes in order, so a class never changes index. `unknown` is not a detector class: images whose only boxes are `unknown` are left out.
- **Splits are per physical part**, so the 5 views of one part never straddle splits.
  - Sources whose train split already has defects (PaintDefect) keep their own split.
  - Anomaly-style sources (Real-IAD, MVTec) have every defect in the official test split. Their defective parts are re-split 70 / 15 / 15 (train / val / test) with a fixed hash. Good test parts stay in test, and 15% of the good training parts move to val.
  - So the detector test set lies **inside the official test set**, which PatchCore never trains on. Both models, and their fusion, can be scored on the same unseen parts.
- **Background images** (no defect): train and val keep 0.25 per defect image (`--background-ratio`); test keeps **every** image of its parts, so pass/fail can be scored on whole parts.
- **Left out:** images labelled defective but with no box (15 in Real-IAD: the mask was empty or only specks). As background they would teach the model that a real defect is normal surface.
- `--task segment` writes polygon labels from the masks (Real-IAD, MVTec) for YOLO-seg; box-only sources such as PaintDefect are refused.

Current build (`polymer` = Real-IAD 512 px, 5 part types + PaintDefect):

| Split | Images | Parts | With defects | Background | Boxes |
|---|---|---|---|---|---|
| train | 7,390 | 2,868 | 5,912 | 1,478 | 6,456 |
| val | 1,452 | 599 | 1,162 | 290 | 1,242 |
| test | 8,191 | 1,690 | 1,192 | 6,999 | 1,257 |

| Boxes per class | train | val | test |
|---|---|---|---|
| contamination | 1,720 | 306 | 311 |
| missing_material | 1,681 | 359 | 390 |
| scratch | 1,206 | 276 | 264 |
| pit_void | 715 | 163 | 168 |
| crack | 438 | 64 | 91 |
| paint_finish | 402 | 38 | 18 |
| deformation | 294 | 36 | 15 |

`deformation` and `paint_finish` come only from PaintDefect; their test sets are small (15 and 18 boxes), so their test numbers are noisy.

## 2. Training (`train`)

Fine-tunes COCO-pretrained weights (`yolo26s.pt` by default; `yolo26n.pt` for quick runs). Augmentation: the Ultralytics defaults (HSV colour and brightness, translate, scale, horizontal flip, mosaic) plus vertical flips and ±10° rotation, because parts can lie in any orientation. Early stop after 30 epochs without val improvement. `--time H` stops after H hours; `--set key=value` passes any other Ultralytics argument. Runs go to `outputs/detector/<name>/` (weights, curves, confusion matrix).

## 3. Evaluation (`eval`)

Writes `<run>/eval/metrics_<split>.json` and `predictions_<split>.csv`:

- **Detection:** mAP50, mAP50-95, and per class: precision, recall, F1, mAP50, mAP50-95.
- **Line metrics** at a confidence threshold (`--conf`, default 0.25): a view fails when any box reaches it.
  - **False-reject rate:** good images or parts that fail. **Escape rate:** defective ones that pass.
  - Per image (with AUROC of the strongest box's confidence), and per part for "fails when ≥ k views fail", k = 1…5. A part with fewer views than k needs all of them. This is the single-view vs multi-view comparison of the ablation study.

## 4. Export (`export`)

ONNX by default (`--format openvino` also works once OpenVINO is installed). The NMS-free head is kept, so the graph outputs final detections, `(1, 300, 6)`: x1, y1, x2, y2, confidence, class. The app needs no NMS code. Measured here: YOLO26n at 512 px runs in **19 ms per view** with ONNX Runtime on this 4-core CPU (plan target: < 150 ms).

## Results

| Run | Where | Data | Model | Epochs | Test mAP50 | Test mAP50-95 |
|---|---|---|---|---|---|---|
| `smoke` | CPU, 3 min | 5% of train | YOLO26n, 512 px | 1 | — (pipeline check only) | — |

Real numbers come from the GPU run in the notebook (YOLO26s, 1024 px).
