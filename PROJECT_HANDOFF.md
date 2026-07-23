# MISFIT Project Handoff

Last updated: 23 July 2026

Repository: <https://github.com/Johesive/MISFIT>

Workspace: `/Users/chenjoh/Documents/MISFIT`
Server root used by notebooks: `/mnt/volume6/czj/labLGN/LabLZ/`

## 1. Read this first

This document is the current handoff for the MISFIT mislocalisation-prediction project. A new task should read this file before editing code, then inspect `xgboost_trial/all_tasks_report.md` for the detailed experiment history.

The old Task 1–6 material inside `all_tasks_report.md` is explicitly historical. Current conclusions and the frozen XGB70 specification are in the report's top sections.

No commit or push has been made for the current group of changes. Preserve all existing user changes and server-only assets, especially `models/`.

## 2. Research question

Primary task: predict whether a human single amino acid variant causes experimentally observed protein mislocalisation.

Primary endpoint:

```python
y = df["Mislocalized"].astype(int)
```

Do not construct the binary target from either of the following:

```python
df["label_5class"].notna()
df["reloc_v3"] > 0
```

Phenotype classes C1–C5 are secondary biological annotations. They are not the source of the primary binary target.

## 3. Current authoritative dataset

Processed table:

```text
data_preparation/cell2024_model_single_subst.csv
```

Current counts:

- 2,179 variants
- 871 genes
- 2,089 main-table rows
- 90 Additional Benign Variants rows
- 1,943 negatives
- 236 positives
- prevalence = 0.1083

Phenotype distribution:

| Class | Meaning | n |
|---|---|---:|
| C0 | no mislocalisation | 1,943 |
| C1 | same-compartment/source-defined positive | 13 |
| C2 | aggregation | 34 |
| C3 | secretory pathway | 121 |
| C4 | nuclear | 29 |
| C5 | cytoplasmic | 39 |
| C1–C5 | positive total | 236 |

Curated correction completed:

```text
RPE65 K294T -> C5_cytoplasmic, Mislocalized = 1
```

Implemented in:

- `data_preparation/3.5.1_classifier_label.ipynb`
- `data_preparation/完整代码.py`

## 4. Canonical identifier and ESM cache issue

Canonical identifier:

```python
KEY = Gene + "||" + Mutation_used
```

All 2,179 canonical keys are unique.

The historical ESM2 pickle is keyed by:

```python
legacy_esm_key = Gene + "||" + Variant
```

Only 88 rows have identical canonical and legacy keys. Directly querying the old cache with `Gene||Mutation_used` previously reduced the feature matrix to 88 rows. The corrected Task 0 uses the legacy key only for old-cache lookup and exports canonical `KEY`. It must fail loudly if the final ESM matrix does not contain all 2,179 rows.

## 5. Frozen primary model and features

Current primary model: **XGBoost 70D (`wt_signal_70`)**, frozen on 23 July 2026. The 64D model is retained as the principal ablation baseline.

| Group | Dimensions |
|---|---:|
| Fold-fitted PCA of ESM2 local delta | 50 |
| Structural | 7 |
| Stability-related | 4 |
| TMD | 3 |
| DeepLoc WT sorting-signal context | 6 |
| **Total** | **70** |

Structural features:

```text
plddt, sasa, rsa, ss_helix, ss_strand, ss_coil,
delta_hydrophobicity
```

Stability-related features:

```text
ddg_esm2, ddg_struct, ddg_rasp, ddg_foldx
```

Availability:

- `ddg_esm2`: 2179/2179
- `ddg_struct`: 2168/2179
- `ddg_rasp`: 2168/2179
- `ddg_foldx`: 2166/2179

TMD model features:

```text
in_TMD
dist_to_nearest_TMD
delta_membrane_insertion
```

Important semantic detail: `dist_to_nearest_TMD` is historically named but actually stores proximity `1/(1+d)`. It equals 1 inside a TMD and approaches 0 with distance.

TMD counts:

- `in_TMD = 1`: 151
- `in_TMD = 0`: 2,028
- positives inside TMD: 39
- positive prevalence inside TMD: 0.2583
- positive prevalence outside TMD: 0.0971

Frozen DeepLoc WT features, generated consistently with Fast/ESM1b:

```text
deeploc_wt_signal_signal_peptide
deeploc_wt_signal_mitochondrial_transit_peptide
deeploc_wt_signal_nuclear_localisation_signal
deeploc_wt_signal_nuclear_export_signal
deeploc_wt_signal_peroxisomal_targeting_signal
deeploc_wt_signal_gpi_anchor
```

These values are externally supervised protein-level context and are shared by variants of the same WT protein. The frozen contract excludes WT localisation probabilities, DeepLoc TM probability, plant-specific signals and all WT–MT DeepLoc delta features.

Frozen classifier:

```python
XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)
```

Training uses balanced sample weights. Any later change to features, hyperparameters, split or random state requires a new model/version name rather than silently replacing XGB70.

The frozen classifier block records the arguments explicitly passed in the completed notebooks. The server runs did not capture complete Python, `xgboost`, `scikit-learn` and CUDA versions, so byte-for-byte reproducibility across environments is not yet guaranteed. Export the server environment and verify prediction checksums before producing a release artefact.

All imputation, scaling and PCA must be fitted on training-fold data only. Splits must be gene-disjoint.

Primary-model registry:

| Status | Model ID | Role | Frozen feature count | Frozen classifier |
|---|---|---|---:|---|
| **Primary** | **XGB70 / `wt_signal_70`** | Current MISFIT predictor | **70** | XGBoost specification above |
| Ablation baseline | XGB64 | Measures the increment from six DeepLoc WT context features | 64 | Same XGBoost specification |
| Historical baseline | XGB61 | Pre-TMD comparison only | 61 | Same XGBoost specification |

## 6. AlphaMissense completion

Notebook:

```text
xgboost_trial/task0.5_complete_alphamissense.ipynb
```

Workflow:

- preserve original scores;
- retrieve missing nonsynonymous annotations through AlphaFold DB metadata `amAnnotationsUrl`;
- do not fabricate scores for synonymous mutations;
- keep explicit retrieval status;
- export canonical `KEY`.

Results:

- original scores: 2,053
- retrieved: 87
- synonymous/not applicable: 31
- no annotation URL: 7
- query error: 1 (`P07203`)
- final available: 2,140/2,179
- paired positives: 235

AlphaMissense-alone paired performance:

- AUROC = 0.6491
- AUPRC = 0.1619

## 7. Completed model results

### Task 12: stability ablation

Notebook: `xgboost_trial/task12_ddg_ablation.ipynb`

Full-cohort five-fold pooled OOF:

| Model | Dimensions | AUROC | AUPRC |
|---|---:|---:|---:|
| PCA + structure | 57 | 0.5898 | 0.1567 |
| + `ddg_esm2` | 58 | 0.6112 | 0.1688 |
| + `ddg_struct`, `ddg_rasp` | 60 | 0.6038 | 0.1697 |
| + `ddg_foldx` | 61 | 0.6286 | 0.1758 |

Paired AlphaMissense cohort:

- AlphaMissense: AUROC 0.6491, AUPRC 0.1619
- MISFIT 61D: AUROC 0.6311, AUPRC 0.1780
- MISFIT − AM: ΔAUROC −0.0179, ΔAUPRC +0.0160

### Task 15: TMD ablation

Notebook: `xgboost_trial/task15_full64.ipynb`

Full-cohort pooled OOF:

| Model | AUROC | AUPRC |
|---|---:|---:|
| 61D | 0.6286 | 0.1758 |
| 64D | 0.6422 | 0.1981 |
| 64D − 61D | +0.0135 | +0.0223 |

Paired AlphaMissense cohort:

- AlphaMissense: 0.6491 / 0.1619
- MISFIT 64D: 0.6442 / 0.1999
- MISFIT − AM: ΔAUROC −0.0048, ΔAUPRC +0.0380

Top DDG/TMD importance ranks in the 64D model:

1. `in_TMD`
2. `ddg_esm2`
3. `dist_to_nearest_TMD`
4. `delta_membrane_insertion`
7. `ddg_rasp`
29. `ddg_struct`
36. `ddg_foldx`

### Task 16: XGBoost vs TabPFN, gene-disjoint 8:1:1

Notebook: `xgboost_trial/task16_tabpfn.ipynb`

Held-out test: n = 210, positives = 23.

| Model | AUROC | AUPRC |
|---|---:|---:|
| XGBoost 64D | 0.6266 | 0.2075 |
| TabPFN 64D | 0.6002 | 0.2441 |
| validation-selected ensemble | 0.6223 | 0.2056 |

The selected TabPFN ensemble weight was 0.60. The ensemble did not outperform XGBoost on test and should not currently be adopted.

Paired AlphaMissense test subset: n = 204, positives = 22.

| Model | AUROC | AUPRC |
|---|---:|---:|
| AlphaMissense | 0.6190 | 0.1552 |
| XGBoost 64D | 0.6391 | 0.2113 |
| TabPFN 64D | 0.6155 | 0.2500 |
| Ensemble | 0.6380 | 0.2096 |

Task 16 is a small single held-out test and cannot replace Task 15 pooled OOF results.

### Task 17: cluster bootstrap and error analysis

Notebook: `xgboost_trial/task17_robustness_error_analysis.ipynb`

Gene-cluster bootstrap used 2,000 replicates on fixed predictions. It estimates evaluation-sample uncertainty conditional on the trained OOF/test predictions. It does not include split, PCA or retraining uncertainty.

Paired differences:

| Comparison | ΔAUROC mean (95% CI) | ΔAUPRC mean (95% CI) |
|---|---:|---:|
| 64D − 61D | +0.0133 (−0.0081, 0.0346) | +0.0229 (−0.0050, 0.0508) |
| 64D − AlphaMissense | −0.0049 (−0.0481, 0.0379) | +0.0388 (−0.0063, 0.0862) |
| TabPFN − XGBoost | −0.0216 (−0.1111, 0.0630) | +0.0394 (−0.0815, 0.1647) |
| Ensemble − XGBoost | −0.0007 (−0.0555, 0.0522) | +0.0008 (−0.0649, 0.0643) |

All paired difference CIs in Task 17 crossed zero. At that historical stage, the evidence did not establish superiority for 64D over 61D, MISFIT over AlphaMissense, or TabPFN/ensemble over XGBoost. Task 17's temporary 64D primary-model decision has since been superseded by Task 18 and the fixed-split XGB70 benchmark.

### Task 18: DeepLoc WT context and full WT–MT experiment

Notebook: `xgboost_trial/task18_deeploc_delta.ipynb`

DeepLoc Fast/ESM1b inference completed for 871 unique WT sequences and 2,148 unique non-WT mutant sequences, covering all 2,179 variants. The main positive result came from six WT sorting-signal probabilities, not WT localisation probabilities or WT–MT deltas.

Full-cohort five-fold pooled OOF:

| Model | AUROC | AUPRC |
|---|---:|---:|
| XGB64 | 0.6422 | 0.1981 |
| **XGB70 (`wt_signal_70`)** | **0.6560** | **0.2479** |

Paired gene-cluster bootstrap, XGB70 − XGB64:

- ΔAUROC +0.0138, 95% CI (−0.0194, +0.0449)
- ΔAUPRC +0.0472, 95% CI (+0.0130, +0.0843)

Paired AlphaMissense cohort (n = 2,140; positives = 235):

| Predictor | AUROC | AUPRC |
|---|---:|---:|
| AlphaMissense | 0.6491 | 0.1619 |
| XGB70 | 0.6579 | 0.2499 |

XGB70 − AlphaMissense fixed-prediction bootstrap:

- ΔAUROC +0.0088, 95% CI (−0.0367, +0.0574)
- ΔAUPRC +0.0866, 95% CI (+0.0271, +0.1475)

The bootstrap is conditional on fixed predictions and does not cover split, retraining or post-hoc model-selection uncertainty. AlphaMissense predicts pathogenicity rather than mislocalisation.

Seven signed WT–MT sorting-signal delta features did not improve the 64D or 70D models. Extreme-delta variants remain available for mechanism-focused analysis, but delta features are excluded from the frozen primary model.

### Fixed 8:1:1 XGB70 and neural-model screening

Fixed gene-disjoint test: n = 210, positives = 23.

| Model | AUROC | AUPRC |
|---|---:|---:|
| XGB64 | 0.6266 | 0.2075 |
| **XGB70** | **0.7012** | **0.2843** |
| MLP64, five-seed ensemble | 0.6661 | 0.2720 |
| MLP70, five-seed ensemble | 0.6536 | 0.2751 |
| FT-Transformer 64D | 0.5820 | 0.1943 |
| FT-Transformer 70D | 0.5406 | 0.2130 |

Paired gene-cluster bootstrap for XGB70 − XGB64 on the fixed test:

- ΔAUROC +0.073, 95% CI (+0.008, +0.140)
- ΔAUPRC +0.072, 95% CI (−0.089, +0.205)

MLP did not show a reliable paired advantage over XGBoost, and FT-Transformer was weaker. These neural routes are not being advanced. The fixed test has been reused for multiple model comparisons and is not an untouched independent validation cohort.

**Current primary modelling approach: frozen XGBoost 70D.** This is a pragmatic selection based on pooled OOF and fixed-test point estimates, stability, efficiency and interpretability; it is not a claim of universally proven superiority.

## 8. TMD robustness and correct interpretation

Subgroup results:

| Subgroup | n | Positive | ΔAUROC 64D−61D | ΔAUPRC 64D−61D |
|---|---:|---:|---:|---:|
| `in_TMD` variants | 151 | 39 | −0.0277 | −0.0514 |
| `in_TMD = 0` variants | 2,028 | 197 | +0.0084 | +0.0143 |

Do not call `in_TMD = 0` rows “non-membrane proteins”. They include both non-membrane proteins and membrane proteins whose tested variant lies outside the annotated TMD.

Current defensible interpretation:

- 64D has a positive overall point estimate, but its CI crosses zero.
- Performance improves in folds 0–2 and declines in folds 3–4 for AUROC.
- The added features do not improve the within-TMD subgroup point estimates.
- It has not been established that the effect is distributed across many genes or is independent of influential genes.
- Feature importance alone does not prove a biological mechanism.

Exploratory OOF threshold analysis used an F1-maximising threshold of 0.277045 chosen on the same OOF predictions:

- TN 1,506
- FP 437
- FN 134
- TP 102
- sensitivity 0.432
- specificity 0.775
- precision 0.189

This threshold analysis is descriptive, not an independent performance estimate.

Important interpretation corrections:

- AlphaMissense predicts general pathogenicity, not mislocalisation.
- A high AlphaMissense score in a MISFIT false positive does not mean AlphaMissense predicts mislocalisation.
- `DNM2||E368K` is a MISFIT high-confidence false negative but has AlphaMissense 0.9869; therefore the two predictors disagree rather than both failing.
- `ADIPOQ||G15G` is a synonymous positive and should be specifically audited.

## 9. TMD audit changes and status

After the initial Task 17 results, `task14_TMD_addition.ipynb` was extended with audit metadata because the original encoding mapped several different states to zero:

- actual non-TMD/non-membrane context;
- UniProt fetch failure;
- no TMD annotation;
- mutation parse failure;
- WT/canonical sequence mismatch;
- position outside sequence.

New audit-only columns:

```text
is_membrane_protein
distance_to_nearest_TMD_residues
tmd_annotation_status
sequence_match
```

These columns are not part of the 64D model and did not change the recorded Task 15/16 results.

Task 14 now saves both:

```text
tmd_features.csv
tmd_features_audited.csv
```

Task 17 prefers the audited file and falls back to the old file. If the updated Task 14 kernel is still alive, only its final save cell needs to run; the long UniProt download cell does not need to be repeated.

The detailed Task 17 output supplied by the user was from the earlier three-column TMD analysis. The new annotation-status stratification has not yet been reported and is optional. Do not invalidate the earlier model metrics because audit columns do not alter model inputs.

## 10. DeepLoc and neural experiments: final status

Task 18 is complete. Raw local DeepLoc assets remain under the ignored `models/deeploc2_package/` directory; derived feature tables, OOF predictions, metrics and bootstrap outputs are tracked under `xgboost_trial/`. Continue to respect the DTU licence before publication, redistribution or commercial use.

Final modelling decisions:

- retain the six WT sorting-signal probabilities in XGB70;
- exclude WT localisation probabilities, DeepLoc TM probability and plant-specific signals;
- exclude signed WT–MT localisation/sorting deltas from the primary model;
- do not advance the current MLP, FT-Transformer, TabPFN or ensemble configurations;
- do not tune further against the reused fixed 8:1:1 test set.

Task 19's current UniProt reviewed/annotation-status analysis is exploratory and does **not** establish exact overlap with the actual DeepLoc training set. Do not use it to claim generalisation to proteins unseen by DeepLoc.

## 11. Files changed in the current uncommitted worktree

Modified:

```text
data_preparation/完整代码.py
xgboost_trial/all_tasks_report.md
```

Untracked/new:

```text
PROJECT_HANDOFF.md
ft_trial/
mlp_trial/
```

This list describes the current local worktree at the time of this handoff update. Do not assume untracked files are disposable. A named pre-pull stash also exists; inspect it before any later recovery or cleanup operation and do not apply it blindly.

No commit or push was performed at the end of this conversation.

## 12. Recommended continuation order

1. Read this handoff and `xgboost_trial/all_tasks_report.md`.
2. Run a read-only Git audit and inspect the current diff before editing.
3. Treat XGB70's 70-feature contract and XGBoost hyperparameters as frozen; record any later change under a new model/version name.
4. Stop using the reused fixed 8:1:1 test set for feature or hyperparameter selection.
5. Audit unusual labels, especially `ADIPOQ||G15G`, before publication claims.
6. Prioritise biological subgroup analysis, error analysis and checks for gene-level concentration or confounding.
7. Seek an external or prospectively held-out cohort for a cleaner generalisation estimate.
8. If a deployable final artefact is required, refit the frozen XGB70 pipeline on all eligible development data and save every preprocessing object; keep that artefact conceptually separate from OOF evaluation predictions.
9. Perform an exact DeepLoc training-set overlap audit only if a valid training-protein list becomes available. UniProt annotation status alone cannot answer that question.
10. Keep the optional TMD audit extension for data-quality analysis; it does not alter the historical 64D inputs or metrics.

## 13. Scientific claim boundary

Current defensible claim:

> Adding six DeepLoc WT sorting-signal probabilities to the 64D variant representation increased the XGBoost pooled-OOF point estimates from 0.6422 to 0.6560 for AUROC and from 0.1981 to 0.2479 for AUPRC. Under a paired gene-cluster bootstrap conditional on the fixed OOF predictions, the AUPRC difference was +0.0472 with a 95% interval of +0.0130 to +0.0843, whereas the AUROC interval crossed zero. XGB70 is therefore the frozen primary model on pragmatic performance, efficiency and interpretability grounds.

This supports a **localisation-aware feature-integration result**: externally supervised WT protein context complements variant-level mutation features. It does not establish a new localisation architecture or a causal sorting-signal mechanism.

Do not currently claim:

- universal or architecture-independent superiority of XGB70;
- superiority over AlphaMissense for pathogenicity or mislocalisation on an independent external cohort;
- that the fixed test is untouched independent validation;
- that the conditional bootstrap covers split, preprocessing, retraining, model-selection or multiplicity uncertainty;
- exact generalisation to proteins absent from the DeepLoc training set;
- a causal sorting-signal mechanism from feature importance or predictive improvement;
- that AlphaMissense directly predicts mislocalisation;
- that TabPFN or the ensemble improves on XGBoost;
- that the reported OOF predictions represent one final model trained on all data.
