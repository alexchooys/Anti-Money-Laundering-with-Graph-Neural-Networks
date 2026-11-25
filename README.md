# AML Temporal Robustness Lab — Deliverable 3

Compare **XGBoost, MLP and GraphSAGE** under frozen, expanding-history,
rolling-window and recency-weighted policies on original Elliptic Bitcoin
transactions. The refined FastAPI interface supports three-seed comparison,
snapshot replay, review-queue export and live frozen-XGBoost prediction.

Author: **Alex (GitHub: alexchooys)**, Applied Data Science, University of Florida.
Contact: https://github.com/alexchooys

## Current evidence

- Executed real-data training/evaluation for seeds 1 and 2, alongside the original
  seed-0 experiment. All runs use train 1–30, validation 31–34 and test 35–49.
- Independent metric recomputation checks **600,120 prediction rows and 540 step
  results**, with identical transaction/target sets across the 36 combinations.
- Eight backend test methods pass, including all 540 replay selections, CSV column
  reordering, real-example prediction, queue export and descriptive input rejection.
- Four PNG/PDF comparison figures are in `docs/figures_d3/`.
- A measured warm-callback comparison verifies equal outputs and quantifies the
  effect of caching. See `results/deliverable3/evaluation/backend_latency_colab.json`
  for the Colab run and `backend_latency.json` for the separate analysis-runtime run.
- The new Colab evidence verifies native inference on 50 real test rows, incoming-degree
  diagnostics, and a same-runtime callback benchmark. Screenshots in `docs/interface_v2/`
  document the new comparison, explorer and live-input layouts.
- Colab median warm-callback latency falls from **55.19 ms to 18.60 ms (2.97×)**.
  This excludes browser/HTTP/network/startup latency and is not a human task-time result.
- `docs/Deliverable_3_Report.pdf` contains the five-page updated report in an IEEE-style
  layout. `docs/Main_IEEEtran.tex` supplies the official IEEE conference-class source
  for Overleaf. The previous report remains in `docs/deliverable2/`.
- No independent human usability study is claimed. Browser batch/download/error
  walkthroughs remain useful refinements, with backend behavior covered by tests.

## Run the refined interface

Python 3.11–3.13. No GPU or raw dataset is required for replay or saved-tree inference.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Recommended for native XGBoost inference:
python -m pip install -r requirements-native.txt
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
```

On Windows use `.venv\Scripts\Activate.ps1`. Open http://127.0.0.1:8000 .
Without native XGBoost, the restricted numeric-tree evaluator supports the provided
checkpoint. Its parity was verified on 50 real feature rows in Deliverable 2.
This is not a general replacement for the XGBoost library.

### Three interface views

1. **Compare experiments:** D2 seed-0 AP beside three-seed mean ± sample SD, plus
   paired recency-minus-expanding gains. These statistics use steps 44–49.
2. **Explore a snapshot:** choose family, policy, seed, time and budget. Inspect
   AP/F1, budget metrics, confusion counts and warnings. Download the full review
   queue. Filtering does not retrain a model or retune a threshold.
3. **Predict from features:** load a real labeled example or enter a 93-number JSON
   array. CSV input accepts 1–200 rows with exactly local_00 through local_92,
   in any order. Download the example CSV to see the format. The app rejects
   missing/duplicate/extra columns, nonnumeric/nonfinite values and oversized
   batches. Download batch predictions after a successful run.

Only the original **frozen XGBoost seed-0 model** serves live predictions, with
threshold 0.70. Scores are uncalibrated and support human review only. Do not apply
the neural scaler to tree-model inputs. Exclude IDs, times, labels and aggregates.
Uploads are not written to disk by the application; hosting platforms may retain
request logs. Do not upload personal banking information to a shared deployment.

## Run in Colab

Open `notebooks/ui_demo_v2.ipynb`, upload the project ZIP when prompted and run the
cells. This does not retrain models. The notebook includes an optional original-data
degree diagnostic, a 50-row real-feature inference check, a same-runtime latency
comparison and a local port-proxy iframe. No public sharing tunnel is required.
The retained execution records are in `docs/colab_v2_evidence/`. New screenshots
are in `docs/interface_v2/`. Instructions are inside the notebook and
`docs/NEXT_STEPS_D3.md`.

## Training and evaluation reproduction

Dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set .
Obtain the three original CSVs from the publisher, observing its reuse terms.
Raw CSVs are not included. The dataset contains 203,769 transactions, 234,355
edges and 49 snapshots; targets include 4,545 illicit, 42,019 licit and 157,205
unknown transactions. Primary predictors are 93 local features, excluding time,
ID and supplied neighborhood aggregates.

- `notebooks/AML_GNN_D3_Training_Only.ipynb` contains the D3 cells extracted with
  their original executed outputs. Use it for D3 reproduction in a GPU Colab runtime.
  Install `requirements-train.txt` first using a compatible PyTorch CUDA build.
  Update the first cell's data/baseline paths and select a **new output directory**.
  Guards deliberately prevent replacement of completed seed runs.
- `notebooks/AML_GNN_Deliverable_3.ipynb` preserves the original combined upload.
  It contains the historical D2 section after the D3 cells. **Do not Run all this
  combined notebook to reproduce D3**, because the old section writes D2 outputs.
- `notebooks/AML_GNN_Real_Data_Setup.ipynb` is the D2 reference for seed 0.
- `notebooks/D3_Results_Backend_Verified.ipynb` records executed audit/API checks
  in the analysis runtime, not a repeat of GPU training or a browser test.
- `src/train.py` retains the D2 frozen-model reproduction CLI. The executed
  notebooks remain the reference for the full adaptation experiment.

Recorded Colab training environment: Python 3.13.16, PyTorch 2.14.1+cu130,
PyG 2.8.1, XGBoost 3.4.1, Tesla T4. Versions and hardware can affect results.

## Recompute saved-result analyses and checks

```bash
python -m src.refinement_audit
python -m src.refinement_figures
python -m unittest discover -s tests -v
python -m src.refinement_benchmark
python -m src.verify_degree_artifacts
# Optional; requires the original dataset:
python -m src.degree_analysis --data-dir data/raw
```

The main analysis averages per-step metrics within each seed, then summarizes
seeds 0, 1 and 2. Error bars show sample SD, not confidence intervals. The optional
degree diagnostic pools transactions within phase/degree groups and labels its AP
accordingly. It uses incoming graph degree, never neighbor labels as predictors.
No window, half-life or threshold is selected using the diagnostic.

## Updated post-shift results

Mean AP across steps 44–49, summarized across three seeds:

| Model | Frozen | Expanding | Rolling | Recency |
|---|---:|---:|---:|---:|
| XGBoost | .0789 ± .0000 | .1983 ± .0000 | .3164 ± .0000 | **.4053 ± .0000** |
| MLP | .1078 ± .0216 | .0656 ± .0075 | .1239 ± .0341 | .1303 ± .0355 |
| GraphSAGE | .0979 ± .0482 | .0793 ± .0057 | .1100 ± .0429 | .1077 ± .0459 |

Sustained recovery is **not observed by step 49 for all 36 combinations** under
the predeclared 90%-of-pre-shift-F1, two-consecutive-supported-step rule. Only
steps 47–48 and 48–49 form supported post-shift pairs. No permanent-failure claim.

XGBoost runs are identical in this full-row/full-feature configuration. A zero
seed SD is not evidence of zero uncertainty on future data. Neural results vary
with initialization, selected checkpoints and thresholds. D3 includes the original
D2 seed-0 run; additional seeds improve the reliability assessment, not the amount
of independent test data. See `docs/DELIVERABLE3_FINDINGS.md` for paired gains.

## Known limitations and responsible use

- Unknown targets are masked, never recoded as licit. Known labels reflect
  investigation priorities and may omit undetected illicit behavior.
- Retrospective snapshots and simulated one-step label delay do not establish
  actual transaction-arrival or investigation-turnaround conditions.
- Initial scaler/class weight stay fixed; per-seed thresholds and neural epoch
  budgets come from validation only. No test-based early stopping or best-seed choice.
- GraphSAGE uses incoming directed edges; architectures have differing capacity.
- Sparse post-shift positives and concentrated validation labels limit conclusions.
- AP is not accuracy. Weighted-model scores are not calibrated probabilities.
- No automatic sanctions, account closure, legal determinations or demonstrated
  financial savings. Anonymized features do not support demographic fairness claims.
- Caches assume immutable result/checkpoint files. Restart the app after replacing
  results. Latency measurements exclude browser, HTTP, network and startup time.

## Project organization

`data/` source instructions; `notebooks/` executed studies and launchers; `src/`
training/data/evaluation/inference; `ui/` app; `results/` artifacts and audits;
`docs/` figures, screenshots and findings; `tests/` behavioral checks.

Preserve the D2 baseline. Commit code, executed notebooks, result artifacts and
screenshots to the existing GitHub repository. Exclude raw CSVs, credentials,
.env, environments and caches. No remote GitHub changes are performed by this package.

## AI assistance

AI tools assisted with design, code generation, auditing, figures and documentation.
The student executed real-data training in Colab. Saved outputs, independent metric
checks and API tests support the reported measurements. Browser evidence, optional
diagnostics and human feedback must reflect actual execution. Follow course disclosure
requirements and review all content before submission.
