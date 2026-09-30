# notebooks

Colab/Kaggle GPU training notebooks (milestone M2). Setup: [docs/SETUP.md](../docs/SETUP.md), step 6.

| Notebook | What it does | Open |
|---|---|---|
| [`train_detector.ipynb`](train_detector.ipynb) | Downloads Real-IAD (1024 px) and PaintDefect, builds the YOLO dataset, trains YOLO26s, evaluates it on the held-out test set and exports ONNX. Needs the `HF_TOKEN` and `ROBOFLOW_API_KEY` secrets. See [docs/DETECTOR.md](../docs/DETECTOR.md). | [Colab](https://colab.research.google.com/github/mescorastudios-creator/ZeroDefect/blob/claude/magical-feynman-bx2o1g/notebooks/train_detector.ipynb) |
