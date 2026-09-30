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

### CPU baseline (2026-09-30)

`cpu_baseline`: YOLO26n, 512 px, trained for 1.5 hours on this 4-core CPU (5 epochs; `--time 1.5 --set close_mosaic=2`) on the `polymer` build. It shows that the pipeline learns; it is not the model to use. The GPU run (YOLO26s, 1024 px, up to 100 epochs) should beat it clearly.

| | Val | Test |
|---|---|---|
| mAP50 | 0.624 | 0.542 |
| mAP50-95 | 0.310 | 0.263 |
| Precision / recall | 0.737 / 0.562 | 0.704 / 0.500 |

| Test, per class | Boxes | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| contamination | 311 | 0.78 | 0.62 | 0.69 | 0.39 |
| missing_material | 390 | 0.67 | 0.64 | 0.65 | 0.33 |
| deformation | 15 | 0.85 | 0.39 | 0.64 | 0.20 |
| crack | 91 | 0.64 | 0.64 | 0.62 | 0.27 |
| scratch | 264 | 0.63 | 0.58 | 0.59 | 0.32 |
| pit_void | 168 | 0.34 | 0.64 | 0.59 | 0.32 |
| paint_finish | 18 | — | 0.00 | 0.02 | 0.01 |

`paint_finish` (paint sags) is not learned yet: the boxes are large, faint regions with vague edges, from only 134 original images, and after 5 epochs the model finds them at confidence ≈ 0.06 only. Labels were checked visually and are correct.

**Pass/fail on the line** (test set, confidence ≥ 0.25). Image-level AUROC of the strongest box: **0.92**.

| | Good | Defective | False-reject rate | Escape rate |
|---|---|---|---|---|
| per image | 6,999 | 1,192 | 6.7% | 25.3% |
| per part, fails if ≥ 1 view fails | 1,258 | 432 | 25.5% | 12.7% |
| per part, ≥ 2 views | 1,258 | 432 | 3.4% | 33.6% |
| per part, ≥ 3 views | 1,258 | 432 | 0.3% | 58.3% |

Real-IAD parts only, at other thresholds (≥ 1 view / ≥ 2 views): confidence 0.1 → false rejects 57% / 18%, escapes 2% / 16%; confidence 0.5 → false rejects 2% / 0%, escapes 22% / 55%.

The multi-view vote trades false rejects against escapes, as expected, but no setting of this baseline is good enough for a line on its own. The next steps are the GPU model and fusion with PatchCore (stage 5), which also sees defect types the detector does not know.

Speed: 32 ms per 512 px image with PyTorch on this CPU; 19 ms with the ONNX export.
