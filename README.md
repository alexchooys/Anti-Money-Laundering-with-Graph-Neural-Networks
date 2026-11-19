# AML Temporal Robustness Lab — Deliverable 2

A working early prototype comparing **XGBoost, MLP and GraphSAGE** under
frozen, expanding-history, rolling-window and recency-weighted policies on
original Elliptic Bitcoin transaction snapshots. The interface supports
saved-experiment inspection and **live frozen-XGBoost inference**.

Author: **Alex (GitHub: alexchooys)**, University of Florida, Applied Data Science.
Contact: https://github.com/alexchooys

## What is actually complete

- The executed Colab training notebook loads real Elliptic data, fits all three
  model families and runs all 12 model-policy combinations for seed 0.
- Real outputs: 180 model-policy-time rows and 200,040 saved prediction rows.
  All saved AP/precision/recall/F1/review-budget metrics were independently
  recomputed from predictions and match (docs/audit.json).
- FastAPI web interface (an equivalent Python web framework to Flask) with
  model/policy/snapshot/budget controls, temporal chart and ranked queue.
- Live inference loads the frozen XGBoost artifact, accepting 93 feature values.
  Native XGBoost is used when installed. A restricted numeric-tree evaluator
  supports lightweight use; parity on 50 real feature rows is verified.
- Interface backend and HTTP tests pass, including all 180 explorer selections,
  low-support warnings, input rejection and a clearly labeled synthetic live probe.

### Verified interface evidence

- `notebooks/ui_demo_executed.ipynb` preserves the successful Colab run.
- `docs/interface_explorer.png` and `docs/interface_live_real.png` show the actual browser interface.
- Fifty real test rows pass native-versus-saved and portable-versus-native inference checks.
  Maximum differences are 2.935e-08 and 5.974e-08 (tolerance 1e-5).
- A real input produces native XGBoost score 0.02013587, below threshold 0.70.
  Details and the real example are in `docs/interface_verification.json`.

Update your GitHub repository and submit its actual URL; this ZIP is not
itself a GitHub repository link. Compile `docs/Main_IEEEtran.tex` in Overleaf
for the official IEEE conference template. The included compiled PDF uses
an IEEE-style two-column layout.

## Quick start: interface (no raw dataset or GPU required)

Python 3.11-3.13 recommended. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
```

On Windows, activate with `.venv\Scripts\Activate.ps1`.
Open http://127.0.0.1:8000 . The default Explorer selection is
XGBoost / recency / step 49 / 10% labeled review budget.

For the native prediction engine, also install `xgboost>=3,<4`.
The zero-vector button is a synthetic software probe, **not an Elliptic
transaction or detection-performance evidence**. Real inputs must be the
93 original local numeric attributes in local_00..local_92 order, excluding
ID, time step, labels and the 72 aggregate attributes. Do not apply the
neural scaler before submitting values to the tree model. Scores are not
calibrated probabilities.

## Colab interface: recommended for your current workflow

Open `notebooks/ui_demo.ipynb` in Colab. It installs the lightweight server,
asks for this project ZIP, verifies a replay/live probe, optionally checks
native parity against original feature rows, and displays the app in a
Colab port-proxy iframe. No public sharing tunnel is required.
The completed run is preserved separately as `ui_demo_executed.ipynb`.
See `docs/NEXT_STEPS.md` for final submission instructions.

## Training and evaluation reproduction

1. Original dataset: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Extract `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`,
   `elliptic_txs_edgelist.csv` into your Drive data folder or `data/raw/`.
   Respect the publisher's reuse terms. No raw data is redistributed.
3. Install `requirements-train.txt`, selecting a compatible CUDA PyTorch
   build for GPU training. Recorded environment: Python 3.13.16,
   PyTorch 2.14.1+cu130, PyG 2.8.1, XGBoost 3.4.1, Tesla T4.
   These are observed versions, not a claim that all systems can install them.
4. Run `notebooks/AML_GNN_Real_Data_Setup.ipynb` in Colab. This is the exact
   executed notebook. Adjust its PROJECT_DIR cell if your Drive path differs;
   it creates its result directory. Its original runtime was already configured
   for PyG; a fresh runtime must install the dependencies first.
5. For one modular frozen-model reproduction:

```bash
python -m src.train --family XGBoost --data-dir data/raw --output-dir results_reproduced --device cuda
# Or --family MLP / GraphSAGE. Use a different output directory per repeated run.
```

The modular CLI is extracted/refactored from the executed notebook and syntax
checked; its full GPU training has not been rerun independently. Use the
executed notebook as the reference for the complete adaptation study. The CLI
and notebook may differ slightly numerically across hardware/library versions.

## Evidence and audit

```bash
python -m unittest discover -s tests -v
python -m src.audit
```

Tests cover backend/HTTP behavior, numeric branch arithmetic and all 180
replay combinations. They are not a substitute for browser screenshots or
native inference parity. `notebooks/interface_backend_verified.ipynb`
contains actual executed backend output. `results/` preserves your uploaded
experiment artifacts. `docs/` contains the report, source and evidence.

## Early results (one seed only)

Mean AP across post-event steps 44-49:

| Model | Frozen | Expanding | Rolling (8 steps) | Recency (half-life 4) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.0789 | 0.1983 | 0.3164 | 0.4053 |
| MLP | 0.0938 | 0.0684 | 0.1179 | 0.1203 |
| GraphSAGE | 0.0716 | 0.0755 | 0.0881 | 0.0817 |

All 12 combinations have sustained F1 recovery **not observed by step 49**
under the predeclared two-consecutive-step/support rule. XGBoost's strong
final-step rebound does not qualify as two-step recovery. Adaptive GraphSAGE
made 41/2/3 post-event alerts for expanding/rolling/recency; all were false
positives at the validation-selected 0.85 threshold. Do not retune on test labels.

## Experimental boundaries and known issues

- Train 1-30, validate 31-34, test 35-49. Step 43 is reported separately.
- Validation contains 508 illicit and 2,481 licit nodes; step 32 contributes
  342 of the 508 illicit labels. Threshold estimates depend strongly on this period.
- Known-label targets only; unknown nodes remain in graph context, not negative targets.
- Fit scaler/class weight on initial training only. One-step label delay is simulated.
- Adaptive model refits use fixed 145/73 epochs for MLP/GraphSAGE, without test stopping.
- Recency weights have mean one; positive weight remains 8.107989 across policies.
- Primary predictors exclude time step and aggregate features. Incoming graph edges
  are retrospective snapshot context; no real-time claim is made.
- Metrics concern selected known labels, not the unknown population. Investigation
  labels can encode blind spots. Seed variation is not a replacement for new time periods.
- Saved prediction replay is available for every experiment. Live neural/graph
  inference and saved adaptive model checkpoints are not yet integrated.
- No automatic sanctions, legal determinations, or measured financial-savings claims.

## Next steps

Commit the updated code, executed demo, screenshots, and report for Deliverable 2. Before
Deliverable 3: seeds 1/2, per-seed validation selection, a label-delay sensitivity
study and UI polish. Keep the predeclared window and half-life fixed; new
settings must not be chosen for favorable holdout results.

## AI assistance

AI tools assisted with design, code generation, documentation, formatting and
audit support. The training notebook was executed on real data in Colab and
its measured outputs are preserved. Backend tests, the executed UI notebook, browser screenshots, and real-feature
parity results are retained as evidence. Review all
content and follow course disclosure requirements.
