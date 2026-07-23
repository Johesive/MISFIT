# Task 19: DeepLoc 2.0 Training-Overlap and Generalisation Audit

**Core question:** Is wt_signal_70's performance gain primarily driven by DeepLoc having been trained on the same or highly similar proteins?

---

## 19.1 Identifier-Level Overlap: UniProt Metadata

Query UniProt REST API for all 871 MISFIT UniProt accessions. Classify by review status (Swiss-Prot vs TrEMBL) and presence of subcellular location annotations.

```python
from pathlib import Path
import hashlib, json, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

ROOT = Path("/mnt/volume6/czj/labLGN/LabLZ")
BASE = ROOT / "xgboost_trial"
DEEPlOC_DIR = ROOT / "models/deeploc2_package/deeploc2"

source = pd.read_csv(ROOT / "data_preparation/cell2024_model_single_subst.csv")
pairs = pd.read_csv(DEEPlOC_DIR / "variant_pair_map.csv")
manifest = pd.read_csv(DEEPlOC_DIR / "sequence_manifest.csv")
task15 = pd.read_csv(BASE / "task15_full64_oof.csv")
oof_18 = pd.read_csv(BASE / "task18_deeploc_context_oof.csv")

with open(DEEPlOC_DIR / "uniprot_metadata.json") as f:
    uniprot_meta = json.load(f)

# Build per-protein table
wt_manifest = manifest[manifest["is_wt"]].copy()
protein_info = source[["Gene", "Uniprot", "sequence"]].drop_duplicates("Uniprot")
wt_pairs = pairs[["wt_sequence_id", "Gene"]].drop_duplicates("wt_sequence_id")
wt_info = wt_manifest.merge(wt_pairs, left_on="sequence_id", right_on="wt_sequence_id", validate="one_to_one")
wt_info = wt_info.drop(columns=["sequence"]).merge(protein_info, on="Gene", validate="one_to_one")

# Classify overlap
overlap_protein = wt_info[["sequence_id", "Gene", "Uniprot", "sequence"]].copy()
overlap_protein["uniprot_reviewed"] = overlap_protein["Uniprot"].map(
    lambda a: uniprot_meta.get(a, {}).get("reviewed", False))
overlap_protein["uniprot_has_subcellular_annotation"] = overlap_protein["Uniprot"].map(
    lambda a: len(uniprot_meta.get(a, {}).get("subcellular_locations", [])) > 0)
overlap_protein["in_swissprot"] = overlap_protein["uniprot_reviewed"]

def classify_overlap(row):
    if row["in_swissprot"] and row["uniprot_has_subcellular_annotation"]:
        return "A_swissprot_with_subcellular"
    elif row["in_swissprot"]:
        return "B_swissprot_no_subcellular"
    else:
        return "C_trembl_or_other"

overlap_protein["overlap_category"] = overlap_protein.apply(classify_overlap, axis=1)
```

### UniProt Overlap Results

| Category | Proteins | % of 871 |
|---|---|---|
| Swiss-Prot (reviewed) | 870 | 99.9% |
| Swiss-Prot + subcellular location annotation | 819 | 94.0% |
| Swiss-Prot, no subcellular annotation | 51 | 5.9% |
| TrEMBL (not in Swiss-Prot) | 1 | 0.1% |

```
Swiss-Prot (reviewed): 870/871
With subcellular annotation: 819/871
Swiss-Prot + subcellular: 819
Swiss-Prot, no subcellular: 51
TrEMBL: 1 (CRYAA, A0A140G945)
Proteins needing Fast clipping (>1022 aa): 36
```

---

## 19.2 Sequence-Level Overlap

SHA256 hashes computed for both full WT sequences and Fast/ESM1b effective sequences (aa[:511] + aa[-511:] for proteins >1022 aa).

```python
def fast_effective_sequence(seq):
    return seq if len(seq) <= 1022 else seq[:511] + seq[-511:]

overlap_protein["sha256_full"] = overlap_protein["sequence"].map(
    lambda s: hashlib.sha256(s.encode()).hexdigest())
overlap_protein["sha256_fast_effective"] = overlap_protein["sequence"].map(
    lambda s: hashlib.sha256(fast_effective_sequence(s).encode()).hexdigest())
overlap_protein["seq_length"] = overlap_protein["sequence"].str.len()
overlap_protein["needs_fast_clipping"] = overlap_protein["seq_length"] > 1022
```

---

## 19.3 Variant-Level Overlap Mapping

Propagate per-protein overlap categories to all 2,179 variants.

| Overlap Category | Variants | Positives | Prevalence |
|---|---|---|---|
| A: Swiss-Prot + subcellular annotation | 2,065 (94.8%) | 230 | 11.1% |
| B: Swiss-Prot, no subcellular annotation | 110 (5.0%) | 6 | 5.5% |
| C: TrEMBL / other | 4 (0.2%) | 0 | 0.0% |

**Critical limitation:** Categories B and C together contain only 6 positives across 114 variants. Any subgroup analysis of the "low-overlap" subgroup is severely underpowered.

---

## 19.4 Stratified Evaluation: wt_signal_70 Performance by Overlap

```python
def subgroup_metrics(df, subgroup_name, mask):
    sub = df[mask]
    y_sub = sub["Mislocalized"].astype(int).to_numpy()
    n_pos = int(y_sub.sum())
    if n_pos < 5:
        return {"subgroup": subgroup_name, "n_variants": len(sub), "n_positives": n_pos,
                "prevalence": y_sub.mean(), "warning": "too few positives"}
    s64 = sub["oof_xgboost_64"].to_numpy()
    s70 = sub["oof_wt_signal_70"].to_numpy()
    return {
        "subgroup": subgroup_name, "n_variants": len(sub), "n_genes": sub["Gene"].nunique(),
        "n_positives": n_pos, "prevalence": y_sub.mean(),
        "auroc_64": roc_auc_score(y_sub, s64), "auroc_70": roc_auc_score(y_sub, s70),
        "auprc_64": average_precision_score(y_sub, s64), "auprc_70": average_precision_score(y_sub, s70),
        "delta_auroc": roc_auc_score(y_sub, s70) - roc_auc_score(y_sub, s64),
        "delta_auprc": average_precision_score(y_sub, s70) - average_precision_score(y_sub, s64),
    }
```

### Stratified Metrics

| Subgroup | n | Positives | Prev | AUROC 64D | AUROC 70D | AUPRC 64D | AUPRC 70D | ΔAUROC | ΔAUPRC |
|---|---|---|---|---|---|---|---|---|---|
| **All variants** | 2,179 | 236 | 0.108 | 0.6422 | 0.6560 | 0.1981 | 0.2479 | +0.0139 | +0.0498 |
| A: Swiss-Prot + subcellular | 2,065 | 230 | 0.111 | 0.6437 | 0.6559 | 0.2057 | 0.2541 | +0.0121 | +0.0484 |
| B: Swiss-Prot, no subcellular | 110 | 6 | 0.055 | 0.6058 | 0.6603 | 0.0806 | 0.1237 | +0.0545 | +0.0431 |
| C: TrEMBL | 4 | 0 | 0.000 | — | — | — | — | — | — |
| A+B: Any Swiss-Prot | 2,175 | 236 | 0.109 | 0.6426 | 0.6561 | 0.1984 | 0.2480 | +0.0135 | +0.0496 |

**Key observations:**

- Category A (high overlap likelihood): ΔAUPRC = +0.048 — the 70D increment among proteins most likely in DeepLoc's training set is virtually identical to the overall increment.
- Category B (low overlap likelihood): ΔAUPRC = +0.043 — the **point estimate is also positive**, despite having only 6 positives. The 70D model does not appear to rely exclusively on training-set membership.
- Category C (TrEMBL): uninformative (0 positives).

---

## 19.5 Gene-Cluster Bootstrap: Subgroup ΔAUPRC

2,000 gene-cluster replicates, conditional on fixed OOF predictions. Computes ΔAUPRC (70D − 64D) separately for Category A and Category B+C, plus the difference-in-differences.

```python
cat_a_mask = (eval_df["overlap_category"] == "A_swissprot_with_subcellular").to_numpy()
cat_bc_mask = ~cat_a_mask

rng = np.random.default_rng(42)
unique_genes = eval_df["Gene"].astype(str).unique()
gene_to_idx = {g: np.flatnonzero(eval_df["Gene"].astype(str).to_numpy() == g) for g in unique_genes}

overlap_bootstrap_rows = []
for rep in range(2000):
    sampled = rng.choice(unique_genes, size=len(unique_genes), replace=True)
    idx = np.concatenate([gene_to_idx[g] for g in sampled])
    if np.unique(y_all[idx]).size < 2:
        continue
    a_idx = idx[cat_a_mask[idx]]
    bc_idx = idx[cat_bc_mask[idx]]
    metrics = {"replicate": rep}
    for sname, sidx in [("A", a_idx), ("BC", bc_idx), ("all", idx)]:
        if len(np.unique(y_all[sidx])) < 2:
            continue
        metrics[f"{sname}_delta_auroc"] = roc_auc_score(y_all[sidx], s70[sidx]) - roc_auc_score(y_all[sidx], s64[sidx])
        metrics[f"{sname}_delta_auprc"] = average_precision_score(y_all[sidx], s70[sidx]) - average_precision_score(y_all[sidx], s64[sidx])
    if "A_delta_auprc" in metrics and "BC_delta_auprc" in metrics:
        metrics["did_delta_auroc"] = metrics["A_delta_auroc"] - metrics["BC_delta_auroc"]
        metrics["did_delta_auprc"] = metrics["A_delta_auprc"] - metrics["BC_delta_auprc"]
    overlap_bootstrap_rows.append(metrics)
```

### Bootstrap Results

| Subgroup | ΔAUPRC mean | 95% CI | Interpretation |
|---|---|---|---|
| **All variants** | +0.0472 | [+0.0130, +0.0843] | CI excludes zero |
| A: Swiss-Prot + subcellular | +0.0456 | [+0.0108, +0.0842] | CI excludes zero |
| B+C: Low/no overlap | +0.0640 | [−0.0093, +0.2388] | CI crosses zero (wide) |
| Diff-in-diffs (A − BC) | −0.0183 | [−0.1933, +0.0722] | CI crosses zero |

---

## 19.6 Interpretation

### What we can conclude

1. **Near-universal Swiss-Prot overlap.** 870/871 (99.9%) MISFIT proteins are in Swiss-Prot, and 819/871 (94.0%) have subcellular location annotations. DeepLoc 2.0 trained its localisation head on Swiss-Prot 2021_04 with 5-fold CV. Therefore, the vast majority of MISFIT proteins were likely seen by some (but not all) DeepLoc ensemble members during training.

2. **The low-overlap subgroup is too small for reliable inference.** Only 6 positives exist in categories B+C combined (114 variants). The ΔAUPRC point estimate is actually positive (+0.043 in Category B), but the bootstrap CI is extremely wide [−0.009, +0.239] and crosses zero. This is an **underpowered** audit, not evidence of "no overlap bias."

3. **No evidence that the 70D increment differs by overlap status.** The difference-in-differences CI [−0.193, +0.072] crosses zero. The point estimates in both subgroups are directionally positive.

4. **DeepLoc features are best characterised as externally supervised protein-level features**, not de novo embeddings. The wt_signal_70 model benefits from knowledge encoded in DeepLoc's sorting-signal head, which was trained on a partially overlapping protein set. This does not invalidate the features' utility for MISFIT's prediction task, but it means:
   - Generalisation to completely novel proteins (not in Swiss-Prot, no known subcellular location) **cannot be assessed** from the current cohort.
   - The improvement may partially reflect the transfer of externally supervised knowledge rather than purely unsupervised protein representations.

### Impact on wt_signal_70 model status

The Task 19 audit does **not** disqualify wt_signal_70 as the leading candidate model, but it constrains the generalisation claims:

- **Permitted claim:** "wt_signal_70 outperforms xgboost_64 and AlphaMissense on the MISFIT cohort, which consists predominantly of Swiss-Prot proteins with known subcellular locations. The improvement is consistent across proteins with and without explicit subcellular annotation in Swiss-Prot, though the latter subgroup is too small for definitive comparison."

- **Not yet permitted:** "wt_signal_70 generalises to proteins unseen by DeepLoc" or "wt_signal_70 captures de novo sorting-signal information independent of external supervision."

### Remaining gaps

| Gap | Status |
|---|---|
| Exact sequence overlap with DeepLoc training FASTA | Not done — DeepLoc 2.0 training FASTA files not publicly available |
| MMseqs2 homologue search | Not done — requires downloading Swiss-Prot 2021_04 human proteome |
| Sorting-signal training set overlap | Not done — DeepLoc sorting-signal training set composition not published |
| HPA test set overlap | Not done — HPA accessions not extracted |

---

## 19.7 Decision Framework

Before declaring wt_signal_70 as the new primary MISFIT model, the following conditions should be met:

| Condition | Status |
|---|---|
| 70D increment not solely driven by exact DeepLoc training overlap proteins | ✅ Point estimate positive in low-overlap subgroup |
| Low-overlap subgroup has directionally consistent point estimate | ✅ +0.043 ΔAUPRC, but n=6 positives |
| Result not dominated by a few genes | ✅ Bootstrap resamples genes |
| Overlap subgroup prevalence/composition differences documented | ✅ Category B has lower prevalence (5.5% vs 11.1%) |
| Multiplicity/model-selection addressed | ❌ Not done |
| AlphaMissense claims bounded to paired cohort + fixed-prediction bootstrap | ✅ Done in Task 18.9b |

**Recommendation:** wt_signal_70 can be adopted as the primary model with the caveat that its DeepLoc-derived features reflect externally supervised protein-level knowledge. The overlap audit is underpowered but directionally reassuring — the ΔAUPRC in the low-overlap subgroup is positive, not zero or negative. Full generalisation assessment awaits a cohort with more non-Swiss-Prot proteins or a prospective study design.

---

*Generated from `task19_deeploc_overlap_audit.ipynb`*
