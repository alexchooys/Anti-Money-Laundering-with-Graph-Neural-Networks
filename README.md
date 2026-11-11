# AML Temporal Robustness Lab

Deliverable 1.2: foundation and design blueprint for a semester project on anti-money-laundering (AML) classification under distribution shift, using the Elliptic Bitcoin transaction graph.

**Research question:** Are graph models more or less robust than tabular and non-graph neural models after the Elliptic step-43 shift, and which model family benefits most from recent-data adaptation?

- **Core models:** XGBoost, MLP (non-graph neural control), GraphSAGE.
- **Core adaptation policies:** frozen, expanding-history retraining, rolling-window retraining, recency-weighted retraining.
- **Scope note:** This is a planned empirical comparison, not a claim of new algorithmic research. Predictions are retrospective classifications of transactions associated with licit or illicit entities. They are not legal findings or production compliance decisions.

## Current status (read before using results)

<!-- UPDATE THIS SECTION after running the real-data notebook. Delete lines that no longer apply. -->

| Item | Status |
|---|---|
| CSV validator, temporal counts, metric helpers, policy helpers, `setup.ipynb` | Implemented |
| Sample (fabricated) fixtures and their setup cells | Executed; for parser testing only, never evidence |
| Real Elliptic data setup (`DATA_MODE = "real"`) | **[TODO: executed on YYYY-MM-DD / not yet executed]** |
| Validation labels, steps 31-34 | **[TODO: N illicit / M licit, from your real run]** |
| Initial frozen XGBoost baseline on real data | **[TODO: run / not yet run]** |
| MLP and GraphSAGE | Architectures defined; training not yet verified locally |
| Streamlit prototype | EDA only; trained-model integration is planned |

No trained AML findings are claimed in this repository. Results in `results/sample/` come from fabricated fixtures. Real outputs are written to `results/real/` only after local execution.

## Installation (macOS / Linux, Python 3.11 recommended)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name aml-gnn --display-name "Python (aml-gnn)"
```

- **Windows PowerShell:** activate with `.venv\Scripts\Activate.ps1`.
- **If PyTorch Geometric fails to install:** install PyTorch first using the [official selector](https://pytorch.org/get-started/locally/), then follow the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), then rerun the requirements install.
- **GPU:** optional. CPU is enough for setup and a small smoke test. Measure memory and runtime before scaling.

## Dataset and access

- **Source:** [Elliptic Data Set on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (a Kaggle account may be required).
- **Download:** manually from Kaggle, or with the Kaggle CLI after configuring your own credentials:
  ```bash
  kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
  ```
- **Place the three CSVs in `data/raw/`** (see `data/raw/README` for expected filenames): the headerless features file, the classes file, and the edge list.
- **Do not commit credentials or the raw data.** `data/raw/` is ignored by Git.
- **Published dataset values (not locally observed):** 203,769 transaction nodes, 234,355 directed edges, 49 time steps; 4,545 illicit / 42,019 licit / 157,205 unknown labels (these sum to the node total).
- **Labels** describe entity-associated transactions, not legal findings. Features are mostly anonymized, which limits explanation and subgroup fairness analysis. Do not attempt re-identification. Respect the publisher's terms before redistributing data.

## How to run

### 1. Smoke test (no download needed)

```bash
jupyter lab setup.ipynb
```

Select the **Python (aml-gnn)** kernel and Run All. The notebook defaults to `DATA_MODE = "sample"`, an explicitly fabricated parser fixture.
**Expected:** all environment checks pass, schema and protocol checks print, and a per-time-step label plot is produced. This only confirms your environment works.

### 2. Real-data check

1. Place the original CSVs in `data/raw/`.
2. In `setup.ipynb`, set `DATA_MODE = "real"`, restart the kernel, and Run All.
3. Save the notebook **with outputs visible**.

**Expected output of a successful real run:**
- Loader totals of 203,769 nodes, 234,355 edges, and 49 time steps, with the label counts above.
- The illicit and licit label counts for validation steps 31-34.
- A per-time-step illicit-count plot.
- All environment checks passing.

If files are missing, the notebook raises an error. It never switches silently from real to sample data.

### 3. Command-line pipeline

```bash
python -m src.data --data-dir data/raw --output-dir results/real
python -m src.baseline --data-dir data/raw --output-dir results/real
python -m streamlit run ui/app.py
```

The baseline selects its threshold only on steps 31-34 and writes metrics for steps 35-49. It is an initial frozen baseline, not the full adaptation experiment.

## Planned evaluation (summary)

- **Split:** train on steps 1-30, validate on 31-34, hold out 35-49. Step 43 is reported separately, with pre-event (35-42) and post-event (44-49) summaries.
- **Inputs:** all core models use the same 93 local numerical features. Time step is metadata only. The 72 supplied neighborhood aggregates form a separate tree ablation.
- **Leakage controls:** preprocessing, hyperparameters, and the threshold are frozen after validation. Each step is scored before its labels are used. Adaptive runs assume a simulated one-step label delay. Unknown labels are excluded from loss and metrics but can supply graph context within an available snapshot. No future snapshots or label features are used.
- **Graph:** primary message passing follows incoming directed edges. Symmetrization is a secondary, retrospective ablation.
- **Metrics:** average precision (AP, not trapezoidal PR-AUC), illicit precision/recall/F1, support counts, and labeled review-budget metrics.
- **Policies:** frozen, expanding history, 8-step rolling window, and recency weighting with a 4-step half-life. These settings are fixed in advance and not tuned on post-shift test results.
- **Non-claims:** no claim of operational savings or real-time compliance readiness.

See `docs/` for the full technical blueprint.

## Workspace structure

```
data/raw/          original CSVs, downloaded separately, ignored by Git
data/sample/       small fabricated schema fixtures, clearly labeled
setup.ipynb        setup, loader, counts, plots, protocol and environment checks
notebooks/         later model and robustness notebooks
src/               data, policies, metrics, initial models, baseline runner
configs/           planned experiment configuration
ui/                Streamlit EDA prototype; trained-inference integration planned
results/sample/    executed fixture output; never empirical evidence
results/real/      generated locally after real-data execution
docs/              blueprint, diagrams, feedback mapping, GitHub guide, checklist
```

## Author and contact

- **Author / maintainer:** Yuanseng Choo (GitHub: [alexchooys](https://github.com/alexchooys))
- **Email:** ychoo@ufl.edu
- **Course:** EEE6778, University of Florida

## References

- Weber et al. (2019), *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.* https://arxiv.org/abs/1908.02591
- PyG `SAGEConv` documentation: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html
