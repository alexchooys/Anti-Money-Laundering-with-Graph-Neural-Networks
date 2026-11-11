# AML Temporal Robustness Lab

**Deliverable 1.2 - foundation and design blueprint.**

Research question: Are graph models more or less robust than tabular and
non-graph neural models after the Elliptic step-43 shift, and which model
family benefits most from recent-data adaptation?

**Core models:** XGBoost, MLP, GraphSAGE. **Core policies:** frozen,
expanding-history retraining, recent-window retraining, recency-weighted retraining.
This is a planned empirical contribution, not a claim of new algorithmic research.

## Current status - read before using results

- Working CSV validator, temporal counts, metric helpers, policy helpers and setup notebook.
- Fabricated software fixtures are bundled and their setup cells were executed.
- No original Elliptic data, real-data counts, or trained AML findings are bundled.
- Initial XGBoost runner and MLP/GraphSAGE architecture definitions are supplied.
  The full PyTorch/PyG/XGBoost/Streamlit stack was not available in the authoring environment;
  those components must be installed and verified locally.
- Streamlit EDA prototype is supplied; trained-model integration is planned.
- Before Canvas submission: run real-data setup and first baseline; create the GitHub
  repository and commit the executed real-data notebook. See docs/SUBMISSION_CHECKLIST.md.

## Installation (macOS / Linux, Python 3.11 recommended)

From this project's root folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name aml-gnn --display-name "Python (aml-gnn)"
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.
For GPU installation, use the official PyTorch selector and compatible PyG guidance:
https://pytorch.org/get-started/locally/ and https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
CPU is adequate for setup and a small smoke test; measure memory/runtime before scaling.

## Dataset and access

Original publisher: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

Download the dataset using your Kaggle account if required and extract the three
CSVs into `data/raw/` (see its README). No credentials belong in this repository.
Expected original dataset: 203,769 nodes, 234,355 edges, 49 time steps,
4,545 illicit / 42,019 licit / 157,205 unknown nodes. These are published values,
not locally observed measurements. Labels describe entity-associated transactions,
not legal findings. Respect publisher terms before redistributing data.

## Run the notebook and real-data check

```bash
jupyter lab setup.ipynb
```

Choose the `Python (aml-gnn)` kernel. The configuration cell deliberately defaults
to `DATA_MODE = "sample"`. This is an explicitly fabricated parser fixture.
After installing the original CSVs, set `DATA_MODE = "real"`, restart the kernel,
and **Run All**. Save with outputs visible. Check the validation illicit count,
the per-time-step label plot, and all environment checks. Missing files raise
an error; there is no silent switch from real to sample data.

```bash
python -m src.data --data-dir data/raw --output-dir results/real
python -m src.baseline --data-dir data/raw --output-dir results/real
python -m streamlit run ui/app.py
```

The baseline selects its threshold only on steps 31-34 and writes metrics for
steps 35-49. It is an initial frozen baseline, not the full adaptation experiment.

## Planned evaluation

- Train 1-30, validate 31-34, hold out 35-49; event step 43 is reported separately.
- All three core models use the same 93 local numerical inputs; time step is
  metadata only. The 72 supplied neighborhood aggregates form a secondary tree ablation.
- Freeze preprocessing, hyperparameters, and threshold after validation. Score each
  time step before using its labels. Adaptive runs assume a one-step label delay.
- Unknown labels are excluded from loss and metrics, but nodes can supply graph
  context inside an available snapshot. No future snapshots or label features.
- Primary graph is incoming directed payment flows. Symmetrization is secondary
  and explicitly retrospective. Fit normalization/class weights on steps 1-30.
- Report AP, illicit precision/recall/F1, sample support and labeled review-budget
  metrics. Average precision is the specified PR summary (not trapezoidal PR-AUC).
- Compare frozen, expanding, 8-step rolling, and 4-step-half-life weighting;
  these fixed settings are not selected using the post-shift test results.
- No claim of operational savings or real-time compliance readiness.

## Workspace structure

- `data/raw/`: original CSVs downloaded separately, ignored by Git.
- `data/sample/`: small fabricated schema fixtures, clearly labeled.
- `setup.ipynb`: setup, loader, counts, plots, protocol checks and environment checks.
- `notebooks/`: later model and robustness notebooks.
- `src/`: data, policies, metrics, initial models and baseline runner.
- `configs/`: planned experiment configuration.
- `ui/`: Streamlit EDA prototype; trained inference integration comes later.
- `results/sample/`: executed fixture output; never empirical evidence.
- `results/real/`: generated locally after real-data execution.
- `docs/`: blueprint, diagrams, feedback mapping, GitHub guide and checklist.

## Author and contact

Author / maintainer: **alexchooys**.
Contact: https://github.com/alexchooys . 
Email: ychoo@ufl.edu

## References

Weber et al. (2019), https://arxiv.org/abs/1908.02591 .
PyG GraphSAGE: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html .
