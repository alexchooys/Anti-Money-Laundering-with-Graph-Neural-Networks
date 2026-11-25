# Deliverable 3: audited early findings

The uploaded notebook contains 85 cells and no saved error outputs. An independent
recalculation reproduces 540 step-metric rows from 600,120 saved predictions.
All experiments use train steps 1–30, validation 31–34 and test 35–49.

## Post-shift AP: mean ± sample SD across three seeds

Mean AP first averages steps 44–49 within each seed.

| Model | Frozen | Expanding | Rolling | Recency |
|---|---:|---:|---:|---:|
| XGBoost | 0.0789 ± 0.0000 | 0.1983 ± 0.0000 | 0.3164 ± 0.0000 | 0.4053 ± 0.0000 |
| MLP | 0.1078 ± 0.0216 | 0.0656 ± 0.0075 | 0.1239 ± 0.0341 | 0.1303 ± 0.0355 |
| GraphSAGE | 0.0979 ± 0.0482 | 0.0793 ± 0.0057 | 0.1100 ± 0.0429 | 0.1077 ± 0.0459 |

Recency minus expanding (paired within seeds):

- GraphSAGE: 0.0284 ± 0.0403 AP.
- MLP: 0.0647 ± 0.0296 AP.
- XGBoost: 0.2070 ± 0.0000 AP.

XGBoost remains the strongest evaluated configuration. Its identical results
across seeds reflect the full-row/full-feature configuration and do not prove
robustness to new data. Neural results vary with initialization and the selected
checkpoint and threshold. Do not select the best seed as the final estimate.

Recovery remains **not observed by step 49 in all 36 combinations**. Only two
consecutive post-shift pairs meet the minimum-support rule. This is short
follow-up, not evidence of permanent failure.

The Deliverable 3 mean includes the Deliverable 2 seed-0 result. Adding seeds
improves assessment of run variability; it is not a new independent test set
or proof of a modeling improvement. AP is not accuracy and these scores are
not calibrated probabilities.

## Implemented interface improvements

- Three-seed comparison table with paired adaptation gains.
- Seed selection, threshold confusion counts and complete review-queue export.
- One-click real example and validated CSV batch input, with downloadable output.
- Missing/duplicate/extra columns, nonnumeric/nonfinite values and oversized batches
  produce explanatory errors. CSV columns align by name.
- Cached file reads and model loading; no raw upload is written to disk by the app.

A Colab paired warm-callback benchmark measured median latency of 55.19 ms for D2
and 18.60 ms for D3 (2.97× speedup), after verifying equal metrics and queue IDs.
Thirty repetitions per version alternate order after three warmups. This excludes
HTTP, browser, network and startup latency and is not a human usability study.
The separate analysis-runtime benchmark is retained without mixing environments.

## Additional verified evidence

Native XGBoost and saved predictions agree on 50 actual test rows, with maximum
absolute difference 2.935e-08. Portable/native differences stay below 5.974e-08.
The v2 real-example score is 0.02013587, below threshold .70. Browser screenshots
show comparison, explorer and input layouts; the input screenshot precedes scoring.

All 216 incoming-degree diagnostic metric rows were recomputed against saved
predictions and the uploaded degree map. For recency after the shift, XGBoost
has higher mean pooled AP than the neural models in each degree group. This
metric pools transactions within phase/group and differs from primary macro AP.
The 1-incoming and 2+-incoming groups contain just 14 and 15 illicit transactions.
Zero incoming degree does not imply isolation. No neighbor labels enter predictors.

## Submission status and remaining limitations

The five-page report, source, updated README, executed training notebook,
results, genuine screenshots and verification evidence are included. Compile
Main_IEEEtran.tex in Overleaf for the exact IEEE conference template before
Canvas submission. The locally compiled PDF uses an IEEE-style layout.
Update the existing GitHub repository and submit its actual URL.

Three seeds cannot correct selective labeling, sparse post-shift positives,
retrospective graph access or short follow-up. No independent human usability
study, universal architecture advantage, significant improvement in detection,
or automatic-sanction capability is claimed.
