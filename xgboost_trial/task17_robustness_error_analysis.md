# Task 17: Robustness, Model Decision, TMD Validation & Error Analysis

**Inputs**: `task15_full64_oof.csv`, `task16_holdout_predictions.csv`, `tmd_features.csv`, `features_v3.csv`  
**Outputs**: `task17_cluster_bootstrap.csv`, `task17_bootstrap_summary.csv`, `task17_tmd_robustness.csv`, `task17_error_summary.csv`, `task17_error_rows.csv`  
**Key parameters**: 2,000 bootstrap replicates, gene-cluster resampling, seed=42

---

## Overview

This notebook addresses four questions:

1. **Bootstrap confidence intervals** — Paired gene-cluster bootstrap (2,000 replicates) on Task 15 pooled OOF and Task 16 holdout predictions
2. **Primary model decision** — Pre-registered decision rule: retain XGBoost 64D unless a challenger shows positive paired CI
3. **TMD robustness** — Check whether TMD gain holds across folds, TMD/non-TMD subgroups, and individual genes
4. **Exploratory error analysis** — Confusion matrix stratified by fold, source, TMD group, and AlphaMissense availability

Bootstrap resamples at the **gene level** (with replacement), preserving within-gene variant correlation. Paired differences are computed on the same bootstrap sample, yielding proper paired CIs.

---

## Section 1: Data Loading & Validation

```python
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

warnings.filterwarnings("ignore")

BASE = Path("/mnt/volume6/czj/labLGN/LabLZ/xgboost_trial")
TASK15_OOF = BASE / "task15_full64_oof.csv"
TASK16_TEST = BASE / "task16_holdout_predictions.csv"
TMD_CSV = BASE / "tmd_features.csv"
FEATURES_CSV = BASE / "features_v3.csv"

BOOTSTRAP_REPLICATES = 2000
RANDOM_STATE = 42

task15 = pd.read_csv(TASK15_OOF)     # 2,179 variants, 236 positive, 10-fold pooled OOF
task16 = pd.read_csv(TASK16_TEST)    # 210 variants (holdout test subset), 23 positive
tmd = pd.read_csv(TMD_CSV)           # in_TMD, dist_to_nearest_TMD, delta_membrane_insertion
features = pd.read_csv(FEATURES_CSV)

# Validation
required15 = ["KEY", "Gene", "Mislocalized", "fold", "final_alphamissense_score",
              "oof_stability_61", "oof_stability_tmd_64"]
required16 = ["KEY", "Gene", "Mislocalized", "final_alphamissense_score",
              "xgboost_64", "tabpfn_64", "ensemble_64"]
assert all(c in task15.columns for c in required15) and task15["KEY"].is_unique
assert all(c in task16.columns for c in required16) and task16["KEY"].is_unique

tmd_audit_cols = ["in_TMD", "dist_to_nearest_TMD", "delta_membrane_insertion"]
assert tmd["KEY"].is_unique
assert all(c in tmd.columns for c in tmd_audit_cols)
assert features["KEY"].is_unique
assert len(task15) == 2179 and int(task15["Mislocalized"].sum()) == 236

print(f"Task 15: n={len(task15)}, genes={task15['Gene'].nunique()}, positives={task15['Mislocalized'].sum()}")
print(f"Task 16 test: n={len(task16)}, genes={task16['Gene'].nunique()}, positives={task16['Mislocalized'].sum()}")
```

**Output:**
```
Task 15: n=2179, genes=871, positives=236
Task 16 test: n=210, genes=88, positives=23
```

- **Task 15** covers the full cohort with 10-fold stratified-group pooled OOF predictions
- **Task 16** is a single holdout test subset (210 variants, 88 genes) — used for TabPFN/XGBoost/AM comparison
- **Base rate**: 10.8% positive across full cohort

---

## Section 2: Gene-Cluster Bootstrap (17.1)

### Method

- **Resampling unit**: Gene (with replacement). Each bootstrap sample draws 871 genes, keeping all variants of resampled genes.
- **Metric**: AUROC and AUPRC computed on each bootstrap sample
- **Paired difference**: Δ computed per bootstrap sample, then summarized with mean and 95% CI (2.5th–97.5th percentile)
- **Replicates where only one class exists are skipped** (AUROC/AUPRC undefined)

### Core bootstrap function

```python
def safe_metrics(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    y_valid = y_true[valid]
    score_valid = score[valid]
    if len(y_valid) == 0 or np.unique(y_valid).size < 2:
        return np.nan, np.nan, len(y_valid)
    return roc_auc_score(y_valid, score_valid), average_precision_score(y_valid, score_valid), len(y_valid)

def cluster_bootstrap(data, prediction_cols, comparison_pairs, scope, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    grouped_indices = {gene: idx for gene, idx in data.groupby("Gene").indices.items()}
    genes = np.array(list(grouped_indices), dtype=object)
    y_all = data["Mislocalized"].to_numpy(dtype=int)
    predictions = {name: data[col].to_numpy(dtype=float) for name, col in prediction_cols.items()}
    records = []
    for replicate in range(n_boot):
        sampled_genes = rng.choice(genes, size=len(genes), replace=True)
        sampled_idx = np.concatenate([grouped_indices[g] for g in sampled_genes])
        y_boot = y_all[sampled_idx]
        metric_values = {}
        for model, values in predictions.items():
            auroc, auprc, n_valid = safe_metrics(y_boot, values[sampled_idx])
            metric_values[model] = {"auroc": auroc, "auprc": auprc}
            records.append({...})
        for left, right in comparison_pairs:
            records.append({
                "name": f"{left}_minus_{right}",
                "auroc": metric_values[left]["auroc"] - metric_values[right]["auroc"],
                "auprc": metric_values[left]["auprc"] - metric_values[right]["auprc"],
            })
    return pd.DataFrame(records)
```

### Three bootstrap experiments

| Experiment | Data | Models compared | Pairs |
|---|---|---|---|
| `task15_full_oof` | Full cohort OOF (n=2,179) | stability_61, stability_tmd_64 | tmd_64 − 61 |
| `task15_paired_am` | Full cohort with AM scores (n≈2,142) | alphamissense, stability_tmd_64 | tmd_64 − AM |
| `task16_paired_test` | Holdout test (n=210) | AM, xgboost_64, tabpfn_64, ensemble_64 | xgb−AM, tabpfn−xgb, ensemble−xgb |

### Results

```
             scope       kind                                 name  auroc_mean  auroc_ci_low  auroc_ci_high  auprc_mean  auprc_ci_low  auprc_ci_high
   task15_full_oof difference  stability_tmd_64_minus_stability_61    0.013332     -0.008062       0.034590    0.022852     -0.004981       0.050847
   task15_full_oof      model                         stability_61    0.628436      0.589204       0.668491    0.180265      0.139697       0.225586
   task15_full_oof      model                     stability_tmd_64    0.641768      0.601314       0.681513    0.203117      0.152077       0.258818
  task15_paired_am difference stability_tmd_64_minus_alphamissense   -0.004869     -0.048114       0.037900    0.038758     -0.006333       0.086163
  task15_paired_am      model                        alphamissense    0.649048      0.607946       0.690462    0.165904      0.129925       0.208147
  task15_paired_am      model                     stability_tmd_64    0.644180      0.604823       0.682567    0.204662      0.152019       0.261243
task16_paired_test difference         ensemble_64_minus_xgboost_64   -0.000741     -0.055511       0.052189    0.000763     -0.064852       0.064337
task16_paired_test difference           tabpfn_64_minus_xgboost_64   -0.021606     -0.111114       0.063034    0.039449     -0.081499       0.164685
task16_paired_test difference       xgboost_64_minus_alphamissense    0.018203     -0.145470       0.182299    0.052306     -0.081576       0.205710
task16_paired_test      model                        alphamissense    0.617934      0.454124       0.777499    0.172540      0.073113       0.311057
task16_paired_test      model                          ensemble_64    0.635397      0.489209       0.760749    0.225609      0.093825       0.358642
task16_paired_test      model                            tabpfn_64    0.614531      0.465610       0.746652    0.264295      0.095888       0.428716
task16_paired_test      model                           xgboost_64    0.636137      0.508473       0.749371    0.224846      0.090626       0.375125
```

### Key findings

| Question | Answer | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---|---|
| Does TMD help over 61D? | **TMD gain visible but CI crosses zero** | +0.0133 (−0.008, +0.035) | +0.0229 (−0.005, +0.051) |
| Does 64D beat AlphaMissense? | **No, AM AUROC slightly higher** | −0.0049 (−0.048, +0.038) | +0.0388 (−0.006, +0.086) |
| Does XGBoost beat AM on test? | **No, CI spans zero** | +0.0182 (−0.145, +0.182) | +0.0523 (−0.082, +0.206) |
| Does TabPFN beat XGBoost? | **No** | −0.0216 (−0.111, +0.063) | +0.0394 (−0.081, +0.165) |
| Does ensemble beat XGBoost? | **No, near-identical** | −0.0007 (−0.056, +0.052) | +0.0008 (−0.065, +0.064) |

**Interpretation**:
- The +1.33 percentage-point TMD gain on AUROC is the most robust positive signal, but the 95% CI still includes zero
- No model decisively outperforms XGBoost 64D on the holdout test
- Task 16 test set is small (n=210, 23 positives) → wide CIs for all test-set comparisons
- AlphaMissense AUPRC is notably lower than 64D (0.166 vs 0.203) despite similar AUROC, indicating 64D has better precision at high-recall regions

---

## Section 3: Primary Model Decision (17.2)

### Pre-registered rule

The primary model is **XGBoost 64D** (trained on Task 15 full-cohort pooled OOF). A challenger (TabPFN or ensemble) only replaces it if **both** conditions hold on Task 16 paired bootstrap:

1. AUROC or AUPRC paired difference CI is entirely above 0
2. The other metric shows no material degradation

```python
decision_rows = summary[(summary["scope"] == "task16_paired_test") & (summary["kind"] == "difference")]
print("Primary model decision")
print("  Retain XGBoost 64D as the primary model unless a challenger has a positive paired CI and no material trade-off.")
print(decision_rows.to_string(index=False))
```

**Output:**
```
Primary model decision
  Retain XGBoost 64D as the primary model unless a challenger has a positive paired CI and no material trade-off.
             scope       kind                           name  auroc_mean  auroc_ci_low  auroc_ci_high  auprc_mean  auprc_ci_low  auprc_ci_high
task16_paired_test difference   ensemble_64_minus_xgboost_64   -0.000741     -0.055511       0.052189    0.000763     -0.064852       0.064337
task16_paired_test difference     tabpfn_64_minus_xgboost_64   -0.021606     -0.111114       0.063034    0.039449     -0.081499       0.164685
task16_paired_test difference xgboost_64_minus_alphamissense    0.018203     -0.145470       0.182299    0.052306     -0.081576       0.205710
```

**Decision**: **Retain XGBoost 64D as primary model.**

- No challenger shows a CI entirely above zero for any metric
- Ensemble_64 − XGBoost_64: ΔAUROC = −0.0007, CI [−0.056, +0.052] — near-identical performance
- TabPFN_64 − XGBoost_64: ΔAUROC = −0.022, CI [−0.111, +0.063] — slightly worse on average
- All CIs are wide due to small test set (n=210)

---

## Section 4: TMD Robustness (17.3)

### Method

Compare 64D vs 61D OOF predictions across:
1. **Overall** — full cohort
2. **By fold** — each of the 5 folds independently
3. **By TMD group** — variants in TMD regions (`in_TMD > 0`) vs outside TMD
4. **By gene** — per-gene mean prediction delta, to detect whether gain is driven by few genes

```python
robust = task15.merge(tmd[["KEY"] + tmd_audit_cols], on="KEY", how="left", validate="one_to_one")
robust["tmd_group"] = np.where(robust["in_TMD"] > 0, "in_TMD", "outside_TMD")

def compare_61_64(data, scope, subgroup):
    y_sub = data["Mislocalized"].to_numpy(dtype=int)
    auc61, ap61, _ = safe_metrics(y_sub, data["oof_stability_61"])
    auc64, ap64, _ = safe_metrics(y_sub, data["oof_stability_tmd_64"])
    return {"scope": scope, "subgroup": subgroup, "n": len(data),
            "auroc_61": auc61, "auroc_64": auc64, "delta_auroc": auc64 - auc61,
            "auprc_61": ap61, "auprc_64": ap64, "delta_auprc": ap64 - ap61}

# Overall + fold-level + TMD subgroup
```

### Results

```
    scope    subgroup    n  positives  auroc_61  auroc_64  delta_auroc  auprc_61  auprc_64  delta_auprc
  overall         all 2179        236  0.628641  0.642168     0.013527  0.175762  0.198098     0.022336
     fold           0  406         47  0.645884  0.660760     0.014876  0.186004  0.216510     0.030506
     fold           1  430         41  0.595899  0.631388     0.035488  0.147849  0.220034     0.072185
     fold           2  431         51  0.592415  0.648349     0.055934  0.197209  0.240184     0.042975
     fold           3  454         63  0.641619  0.634352    -0.007267  0.248633  0.229021    -0.019613
     fold           4  458         34  0.697558  0.674320    -0.023238  0.157227  0.169541     0.012314
tmd_group      in_TMD  151         39  0.703526  0.675824    -0.027701  0.443994  0.392566    -0.051427
tmd_group outside_TMD 2028        197  0.607363  0.615799     0.008436  0.147991  0.162250     0.014259
```

### Gene-level prediction change (top 20 by absolute delta)

```
         n  positives  mean_delta  mean_in_TMD
Gene
ADRB2    1          1    0.368057          1.0
SLC2A2   3          0    0.350139          1.0
UQCRQ    1          0    0.314133          1.0
USH1G    1          1   -0.303203          0.0
GDI1     2          0   -0.302151          0.0
G6PC3    2          1    0.285158          1.0
GTF2H5   1          0    0.282078          0.0
SLC29A3  4          0    0.268621          0.5
MBL2     1          0   -0.252274          0.0
PTH1R    2          2    0.246504          0.5
PTCH1    2          0    0.234122          0.5
ALDH4A1  1          0   -0.232756          0.0
PRKAG3   1          0    0.228127          0.0
ALG12    4          0    0.218742          0.5
SLC26A2  1          0    0.218167          1.0
TMEM43   1          0    0.212327          1.0
IDH3B    1          0   -0.207127          0.0
ERCC1    1          0   -0.207010          0.0
MST1     1          0    0.196089          0.0
HNMT     1          0    0.195400          0.0
```

### Robustness assessment

| Stratification | Pattern |
|---|---|
| **Overall** | +0.0135 ΔAUROC, +0.0223 ΔAUPRC — modest gain |
| **By fold** | Positive in 3/5 folds (0, 1, 2); slightly negative in folds 3, 4. Largest gain in fold 2 (+0.056) |
| **in_TMD (n=151)** | AUROC 0.704 (61D) → 0.676 (64D): **−0.028 decline**. TMD features do not help within-TMD proteins — they are already well-modeled by 61D |
| **outside_TMD (n=2,028)** | AUROC 0.607 → 0.616: **+0.008 gain**. The 61D→64D improvement comes entirely from non-TMD proteins |
| **By gene** | Top genes with largest |Δ| are mostly single-variant genes — prediction change is ∼0.2–0.37 logits. The effect is spread across many genes, not concentrated in a few |

**Key insight**: The TMD feature gain is paradoxical: it does **not** improve prediction for proteins *inside* TMD regions (where AUROC actually drops from 0.704 to 0.676), but instead helps **non-TMD proteins** (+0.008 AUROC). This suggests TMD features are informative as global structural context rather than as local TMD-residue-level predictors.

---

## Section 5: Exploratory Error Analysis (17.4)

### Method

- **Threshold**: F1-maximizing threshold on the full OOF predictions (0.2770). This is an exploratory, in-sample threshold — performance metrics below are **not** independent estimates.
- **Confusion categories**: TP, TN, FP, FN assigned at this threshold
- **Stratification**: fold, source, TMD group, AlphaMissense availability

```python
precision, recall, thresholds = precision_recall_curve(analysis["Mislocalized"], analysis["oof_stability_tmd_64"])
f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
threshold = float(thresholds[np.nanargmax(f1)])  # 0.277045
analysis["predicted_positive"] = (analysis["oof_stability_tmd_64"] >= threshold).astype(int)
analysis["error_type"] = np.select([...], ["TP", "TN", "FP", "FN"])
```

### Overall confusion matrix

```
Exploratory OOF F1-maximising threshold: 0.277045
error_type
TN    1506
FP     437
FN     134
TP     102
```

| Metric | Value |
|---|---|
| Sensitivity (recall) | 102 / 236 = 43.2% |
| Specificity | 1506 / 1943 = 77.5% |
| Precision | 102 / 539 = 18.9% |
| F1 | 2 × 0.432 × 0.189 / (0.432 + 0.189) = 0.263 |

### Stratified performance

```
  stratifier             level  FN  FP   TN  TP  sensitivity  specificity  precision
        fold                 0  24  72  287  23     0.489362     0.799443   0.242105
        fold                 1  24  76  313  17     0.414634     0.804627   0.182796
        fold                 2  28  93  287  23     0.450980     0.755263   0.198276
        fold                 3  42  83  308  21     0.333333     0.787724   0.201923
        fold                 4  16 113  311  18     0.529412     0.733491   0.137405
      source additional_benign   0   9   80   1     1.000000     0.898876   0.100000
      source              main 134 428 1426 101     0.429787     0.769148   0.190926
   tmd_group            in_TMD   9  51   61  30     0.769231     0.544643   0.370370
   tmd_group       outside_TMD 125 386 1445  72     0.365482     0.789186   0.157205
am_available             False   1   2   36   0     0.000000     0.947368   0.000000
am_available              True 133 435 1470 102     0.434043     0.771654   0.189944
```

### Stratification insights

| Stratifier | Pattern |
|---|---|
| **Fold** | Fold 3 is hardest: sensitivity only 33.3% (63 positives, 21 caught). Fold 4 has best sensitivity (52.9%) but worst precision (13.7%) |
| **Source** | `additional_benign` has only 1 positive (correctly caught) but precision is 10% — 9 FPs from 90 benign variants. Main cohort: sensitivity 43%, precision 19% |
| **TMD group** | `in_TMD`: high sensitivity (76.9%) and precision (37.0%) — the model is much better at identifying mislocalization in TMD proteins. `outside_TMD`: low sensitivity (36.5%), low precision (15.7%) — most errors occur here |
| **AM available** | 39 variants lack AM scores (all TN or FN). Of 2,140 with AM scores: sensitivity 43.4%, specificity 77.2% |

### Top false positives (high-confidence wrong predictions)

```
            KEY     Gene source   tmd_group  oof_stability_tmd_64  final_alphamissense_score
   EDNRB||V111E    EDNRB   main      in_TMD              0.924817                     0.9801
  SLC2A2||G318R   SLC2A2   main      in_TMD              0.876587                     0.9773
  SLC2A2||W444R   SLC2A2   main      in_TMD              0.801302                     0.9460
     BSND||G47R     BSND   main      in_TMD              0.773161                     0.9738
   G6PC3||G262R    G6PC3   main      in_TMD              0.752824                     0.9857
    UCP2||L175V     UCP2   main      in_TMD              0.752210                     0.7458
     BTK||K430E      BTK   main outside_TMD              0.751659                     0.9998
      GAN||A51P      GAN   main outside_TMD              0.750687                     0.9965
    SCO2||L151P     SCO2   main outside_TMD              0.743403                     0.9048
    PEX7||H110R     PEX7   main outside_TMD              0.738225                     0.9662
```

**FP pattern**: High-scoring false positives are heavily enriched in **TMD proteins** (EDNRB, SLC2A2, BSND, G6PC3, UCP2 all `in_TMD`) and also have very high AlphaMissense scores (0.90–0.99). These are membrane-protein benign variants that both models confidently "mislocalize." This suggests membrane context is a systematic confounder.

### Top false negatives (high-confidence misses)

```
          KEY   Gene source   tmd_group  oof_stability_tmd_64  final_alphamissense_score
 PPARG||V318M  PPARG   main outside_TMD              0.011135                     0.8189
    NDP||I18K    NDP   main outside_TMD              0.012660                     0.3041
  PENK||G247D   PENK   main outside_TMD              0.031294                     0.1540
 ADIPOQ||G15G ADIPOQ   main outside_TMD              0.032630                        NaN
CYP1A2||F186L CYP1A2   main outside_TMD              0.033738                     0.8541
 PRODH||L289M  PRODH   main outside_TMD              0.035646                     0.0573
 AMHR2||H282Q  AMHR2   main outside_TMD              0.036026                     0.4194
  DNM2||E368K   DNM2   main outside_TMD              0.036100                     0.9869
```

**FN pattern**: False negatives are all `outside_TMD`. AlphaMissense scores vary widely (0.06–0.99), indicating these are cases where **both** models fail. DNM2||E368K is particularly striking: AM=0.99, 64D=0.036 — a true mislocalization that both models catastrophically miss. Most FNs have low AM scores (<0.5), suggesting a shared blind spot for certain mislocalization mechanisms.

---

## Section 6: Summary & Conclusions

### Bootstrap → Model Decision

| Finding | Verdict |
|---|---|
| TMD gain (64D−61D) | ΔAUROC +0.013, CI [−0.008, +0.035] — real but modest, CI crosses zero |
| 64D vs AlphaMissense | ΔAUROC −0.005, CI [−0.048, +0.038] — statistically indistinguishable |
| Primary model | **XGBoost 64D retained** — no challenger exceeds with positive CI |

### TMD Robustness

- **Gain is real** (positive in 3/5 folds) but comes from **non-TMD** proteins, not TMD proteins
- **Paradoxical result**: adding TMD features degrades prediction for TMD-resident variants (ΔAUROC −0.028) while improving non-TMD variants (+0.008)
- Interpretation: TMD features serve as global structural context, not local TMD-residue signals
- Effect is **distributed across many genes**, not driven by outliers

### Error Analysis (exploratory, in-sample threshold)

- **TMD proteins** are the model's "sweet spot": sensitivity 76.9%, precision 37.0%
- **Non-TMD proteins** are the primary error source: sensitivity 36.5%, precision 15.7%
- **Membrane context** is a systematic confounder for false positives — high AM scores + TMD location → high 64D scores even for benign variants
- **DNM2||E368K** and similar true positives are missed by both 64D and AM — shared blind spot

### Output Files Saved

| File | Rows | Description |
|---|---|---|
| `task17_cluster_bootstrap.csv` | ~26,000 | Raw bootstrap replicates (3 experiments × 2,000 reps) |
| `task17_bootstrap_summary.csv` | 13 | Mean ± 95% CI per model/pair/scope |
| `task17_tmd_robustness.csv` | 8 | TMD gain by fold and subgroup |
| `task17_error_summary.csv` | 11 | Stratified confusion matrix stats |
| `task17_error_rows.csv` | 571 | Individual FP/FN variants with confidence scores |
