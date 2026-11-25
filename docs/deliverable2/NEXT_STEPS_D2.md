# Deliverable 2: final submission steps

Real-data training, seed-0 evaluation, browser screenshots, and inference
parity on 50 actual test transactions are complete. No additional experiment
is required to document the current prototype.

1. Extract AML_GNN_Deliverable_2_Updated.zip. Put the CONTENTS of aml-gnn
   at your repository root. Preserve any unrelated existing work.
2. Review README.md author/contact and AI assistance disclosure.
3. For the exact required IEEE format, upload IEEE_Report_Source.zip to
   Overleaf, select Main_IEEEtran.tex as the main document and compile.
   The included locally compiled report has the same content in an IEEE-style
   two-column layout. Check that the IEEEtran output remains 4-6 pages.
4. Commit code, executed notebooks, results, docs/screenshots and README.
   Exclude raw data, .env, credentials, .venv and caches. Browser uploads do
   not apply .gitignore automatically. No remote GitHub write is performed here.
5. Submit the updated repository URL and final IEEE PDF to Canvas.

Suggested Git commands in an existing local clone:

```bash
git status
git add README.md requirements.txt requirements-train.txt .gitignore src ui notebooks results docs data tests
git diff --cached --stat
git commit -m "Verify AML prototype on real data and add interface evidence"
git push
```

Recorded interface evidence:
- notebooks/ui_demo_executed.ipynb: no saved errors.
- docs/interface_explorer.png: XGBoost/recency/49, 10% review budget.
- docs/interface_live_real.png: real-feature native XGBoost score 0.02013587.
- docs/interface_verification.json: 50-row parity checks and checkpoint hash.

Keep results preliminary: one seed, selective labels, short follow-up,
and no observed sustained recovery by step 49. Extra seeds and diagnostic
analysis are refinements for Deliverable 3, not completed findings.
