# ScribblBot

Real time sketch recognition powered by a lightweight CNN trained on Google's Quick Draw dataset. Draw anything in the browser and ScribblBot identifies it instantly.

**Live app:** https://huggingface.co/spaces/Jog-sama/ScribbleBot_Huggingface

---

## Results

| Model | Architecture | Test Accuracy |
|---|---|---|
| Majority Class Baseline | Always predicts most frequent class | 6.67% |
| Random Forest | HOG features, 200 trees | 85.10% |
| ScribblNet | 3-layer CNN | **94.42%** |

ScribblNet was trained on 30,000 samples across 15 classes (2,000 per class) for 15 epochs using Adam with cosine annealing. Training took under 5 minutes on Apple M-series hardware with MPS acceleration. The Random Forest operates on 1,296-dimensional HOG feature vectors extracted from 28×28 grayscale bitmaps.

**Classes:** cat · dog · pizza · bicycle · house · sun · tree · car · fish · butterfly · guitar · hamburger · airplane · banana · star

### Per-class performance (ScribblNet)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| cat | 0.88 | 0.83 | 0.85 |
| dog | 0.81 | 0.82 | 0.82 |
| pizza | 0.94 | 0.97 | 0.96 |
| bicycle | 0.96 | 0.98 | 0.97 |
| house | 0.99 | 0.98 | 0.98 |
| sun | 0.95 | 0.96 | 0.96 |
| tree | 0.96 | 0.97 | 0.97 |
| car | 0.97 | 0.96 | 0.96 |
| fish | 0.97 | 0.95 | 0.96 |
| butterfly | 0.97 | 0.97 | 0.97 |
| guitar | 0.95 | 0.98 | 0.96 |
| hamburger | 0.99 | 0.97 | 0.98 |
| airplane | 0.91 | 0.89 | 0.90 |
| banana | 0.97 | 0.98 | 0.97 |
| star | 0.94 | 0.94 | 0.94 |

Cat and dog are the hardest classes, which is expected given their visual similarity in quick sketches. Airplane also underperforms, likely due to style variation in how people draw wings and fuselage.

---

## Experiment: Training Size Sensitivity

Both ScribblNet and Random Forest were trained at 10%, 25%, 50%, 75%, and 100% of available training data.

| Fraction | Samples | ScribblNet | Random Forest |
|---|---|---|---|
| 10% | 3,000 | 86.12% | 77.92% |
| 25% | 7,500 | 90.37% | 80.35% |
| 50% | 15,000 | 92.70% | 81.57% |
| 75% | 22,500 | 93.00% | 82.88% |
| 100% | 30,000 | 94.02% | 83.03% |

The CNN scales more steeply with data volume than the Random Forest. At 10% of training data the gap is about 8 points; at 100% it grows to 11 points. The Random Forest plateaus around 83% while ScribblNet continues improving, suggesting the CNN would benefit further from additional data.

---

## Dataset

[Quick Draw](https://quickdraw.withgoogle.com/data) by Google — 50 million drawings across 345 categories, collected from players of the Quick Draw game. Each drawing is a 28×28 grayscale bitmap stored as a flat 784-element uint8 vector. The dataset is publicly available via Google Cloud Storage.

---

## Setup

```bash
pip install -r requirements.txt
python setup.py
python app.py
```

`setup.py` runs the full pipeline: downloads the raw `.npy` files, extracts HOG features, trains all three models, and runs the experiment.

Individual steps:

```bash
python scripts/make_dataset.py
python scripts/build_features.py
python scripts/model.py
```

---

## Repository Structure

```
scribblbot/
├── README.md
├── requirements.txt
├── Makefile
├── setup.py
├── app.py
├── config.py
├── scripts/
│   ├── make_dataset.py
│   ├── build_features.py
│   └── model.py
├── models/
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
└── notebooks/
```

| Component | Location |
|---|---|
| Naive baseline | `scripts/model.py` — `MajorityClassifier`, saved to `models/naive_model.pkl` |
| Random Forest | `scripts/model.py` — `train_classical()`, saved to `models/classical_model.pkl` |
| ScribblNet CNN | `scripts/model.py` — `ScribblNet`, `train_deep()`, saved to `models/deep_model.pth` |
| Inference app | `app.py` |
| Config | `config.py` |

---

## Deployment

1. `python setup.py` to train and generate `models/deep_model.pth`
2. Create a new Space on HuggingFace (SDK: Gradio)
3. Push the full repo including `models/deep_model.pth`

---

## Git Workflow

Working branches: `develop` for integration, `feature/*` for individual changes. All work branches into `develop` via pull requests. `develop` merges into `main` for releases. No direct commits to `main`.
