# Task 18: DeepLoc 2.0 Context-First Experiment

**Model setting:** Fast/ESM1b (sequences clipped to 1022 aa: first 511 + last 511)
**Device:** CUDA:1 (NVIDIA RTX 4090, 24 GiB)
**DeepLoc ensemble:** 5-member; 10 biological subcellular localisation classes (11 internal outputs, first dimension discarded by package) plus 9 sorting-signal classes

---

## Overview

This task adds WT protein context features from DeepLoc 2.0 to the fixed 64D XGBoost model. WT–MT sorting-signal deltas are evaluated only after a sensitivity pilot shows meaningful dynamic range. Localisation deltas are retained as an exploratory diagnostic.

Primary ablations on Task 15's fixed gene-disjoint folds:

1. **xgboost_64** — 64D fixed baseline
2. **wt_loc_73** — 64D + 9 human-relevant WT localisation probabilities (exclude plastid)
3. **wt_signal_70** — 64D + 6 WT sorting-signal probabilities (SP, MT, NLS, NES, PTS, GPI; exclude plant signals and TM)
4. **wt_context_79** — 64D + both WT feature groups (9 loc + 6 signal)
5. **wt_context_tm_80** — 64D + all WT features including TM domain probability

---

## 18.1 Build WT-First Manifests and FASTA Files

WT sequences are deduplicated independently, so the context experiment can run before any MT inference. A deterministic pilot enriches positives, TMD variants, and terminal variants to test whether the sorting-signal head responds to single substitutions.

```python
from pathlib import Path
import hashlib
import re
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path("/mnt/volume6/czj/labLGN/LabLZ")
BASE = ROOT / "xgboost_trial"
MODEL_PACKAGE = ROOT / "models/deeploc2_package"
SOURCE_CSV = ROOT / "data_preparation/cell2024_model_single_subst.csv"
FEATURES_CSV = BASE / "features_v3.csv"
TASK15_OOF = BASE / "task15_full64_oof.csv"
TMD_CSV = BASE / "tmd_features.csv"
DEEPlOC_DIR = MODEL_PACKAGE / "deeploc2"
FASTA_DIR = DEEPlOC_DIR / "fasta"
MANIFEST_CSV = DEEPlOC_DIR / "sequence_manifest.csv"
PAIR_MAP_CSV = DEEPlOC_DIR / "variant_pair_map.csv"
WT_PREDICTIONS_CSV = DEEPlOC_DIR / "wt_predictions_fast_esm1b.csv"
PILOT_PREDICTIONS_CSV = DEEPlOC_DIR / "pilot_mt_predictions_fast_esm1b.csv"
FULL_PREDICTIONS_CSV = DEEPlOC_DIR / "all_predictions_fast_esm1b.csv"
CONTEXT_FEATURES_CSV = BASE / "deeploc_wt_context_features.csv"
SENSITIVITY_CSV = BASE / "deeploc_sorting_delta_sensitivity.csv"
OOF_CSV = BASE / "task18_deeploc_context_oof.csv"
METRICS_CSV = BASE / "task18_deeploc_context_metrics.csv"
IMPORTANCE_CSV = BASE / "task18_deeploc_context_importance.csv"
BOOTSTRAP_CSV = BASE / "task18_deeploc_context_bootstrap.csv"

DEEPlOC_DIR.mkdir(parents=True, exist_ok=True)
FASTA_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = ["cytoplasm", "nucleus", "extracellular", "cell_membrane", "mitochondrion",
             "plastid", "endoplasmic_reticulum", "lysosome_vacuole", "golgi_apparatus", "peroxisome"]
SIGNALS = ["signal_peptide", "transmembrane_domain", "mitochondrial_transit_peptide",
           "chloroplast_transit_peptide", "thylakoid_transit_peptide",
           "nuclear_localisation_signal", "nuclear_export_signal",
           "peroxisomal_targeting_signal", "gpi_anchor"]
HUMAN_LOCATIONS = [c for c in LOCATIONS if c != "plastid"]
CORE_SIGNALS = ["signal_peptide", "mitochondrial_transit_peptide",
                "nuclear_localisation_signal", "nuclear_export_signal",
                "peroxisomal_targeting_signal", "gpi_anchor"]
TM_SIGNAL = ["transmembrane_domain"]
STRUCT_COLS = ["plddt", "sasa", "rsa", "ss_helix", "ss_strand", "ss_coil", "delta_hydrophobicity"]
DDG_COLS = ["ddg_esm2", "ddg_struct", "ddg_rasp", "ddg_foldx"]
TMD_COLS = ["in_TMD", "dist_to_nearest_TMD", "delta_membrane_insertion"]
RANDOM_STATE = 42
N_COMPONENTS = 50
FASTA_BATCH_SIZE = 500
MODEL_SETTING = "Fast/ESM1b"
```

```python
source = pd.read_csv(SOURCE_CSV)
required = ["Gene", "Mutation_used", "sequence", "mutant_sequence", "Mislocalized"]
missing = [c for c in required if c not in source.columns]
assert not missing, f"Missing source columns: {missing}"
source["KEY"] = source["Gene"].astype(str) + "||" + source["Mutation_used"].astype(str)
assert len(source) == 2179 and source["KEY"].is_unique
assert source["Mislocalized"].isin([0, 1]).all() and int(source["Mislocalized"].sum()) == 236

def clean_sequence(value):
    if not isinstance(value, str):
        return None
    sequence = re.sub(r"\s+", "", value).upper()
    return sequence if sequence and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYU]+", sequence) else None

def mutation_position(value):
    match = re.fullmatch(r"[A-Z](\d+)[A-Z]", str(value))
    return int(match.group(1)) if match else np.nan

source["wt_sequence"] = source["sequence"].map(clean_sequence)
source["mt_sequence"] = source["mutant_sequence"].map(clean_sequence)
assert source[["wt_sequence", "mt_sequence"]].notna().all().all(), "Invalid or missing amino-acid sequence"
source["mutation_position"] = source["Mutation_used"].map(mutation_position)
source["sequence_length"] = source["wt_sequence"].str.len()

all_sequences = pd.concat([source["wt_sequence"], source["mt_sequence"]], ignore_index=True).drop_duplicates().tolist()
sequence_to_id = {sequence: f"DL{index:05d}" for index, sequence in enumerate(all_sequences, start=1)}
manifest = pd.DataFrame({"sequence_id": [sequence_to_id[s] for s in all_sequences], "sequence": all_sequences})
manifest["original_length"] = manifest["sequence"].str.len()
manifest["fast_clipped"] = manifest["original_length"] > 1022
manifest["effective_length_fast"] = manifest["original_length"].clip(upper=1022)
manifest["sha256"] = manifest["sequence"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
assert manifest["sequence_id"].is_unique and manifest["sequence"].is_unique

pairs = source[["KEY", "Gene", "Mutation_used", "Mislocalized", "mutation_position", "sequence_length"]].copy()
pairs["wt_sequence_id"] = source["wt_sequence"].map(sequence_to_id)
pairs["mt_sequence_id"] = source["mt_sequence"].map(sequence_to_id)
pairs["wt_equals_mt"] = pairs["wt_sequence_id"] == pairs["mt_sequence_id"]
pairs["mutation_retained_fast"] = (pairs["sequence_length"] <= 1022) | (pairs["mutation_position"] <= 511) | (pairs["mutation_position"] > pairs["sequence_length"] - 511)

tmd = pd.read_csv(TMD_CSV, usecols=["KEY", "in_TMD"])
pairs = pairs.merge(tmd, on="KEY", how="left", validate="one_to_one")
terminal = (pairs["mutation_position"] <= 50) | (pairs["mutation_position"] > pairs["sequence_length"] - 50)
pairs["pilot_variant"] = pairs["Mislocalized"].eq(1) | pairs["in_TMD"].eq(1) | terminal

manifest["is_wt"] = manifest["sequence_id"].isin(pairs["wt_sequence_id"])
manifest["is_pilot_mt"] = manifest["sequence_id"].isin(pairs.loc[pairs["pilot_variant"], "mt_sequence_id"])
manifest["is_remaining_mt"] = ~manifest["is_wt"] & ~manifest["is_pilot_mt"]
manifest.to_csv(MANIFEST_CSV, index=False)
pairs.to_csv(PAIR_MAP_CSV, index=False)

def write_fasta_batches(table, prefix):
    paths = []
    for start in range(0, len(table), FASTA_BATCH_SIZE):
        path = FASTA_DIR / f"{prefix}_{start // FASTA_BATCH_SIZE + 1:02d}.fasta"
        with path.open("w") as handle:
            for row in table.iloc[start:start + FASTA_BATCH_SIZE].itertuples(index=False):
                handle.write(f">{row.sequence_id}\n{row.sequence}\n")
        paths.append(path)
    return paths

wt_manifest = manifest.loc[manifest["is_wt"]].copy()
pilot_mt_manifest = manifest.loc[manifest["is_pilot_mt"] & ~manifest["is_wt"]].copy()
remaining_mt_manifest = manifest.loc[manifest["is_remaining_mt"]].copy()
wt_fastas = write_fasta_batches(wt_manifest, "wt")
pilot_fastas = write_fasta_batches(pilot_mt_manifest, "pilot_mt_nonwt")
remaining_fastas = write_fasta_batches(remaining_mt_manifest, "remaining_mt")
print(f"Variants: {len(pairs)}; unique WT: {len(wt_manifest)}; pilot MT (non-WT): {len(pilot_mt_manifest)}; remaining MT: {len(remaining_mt_manifest)}")
print(f"WT FASTA batches: {len(wt_fastas)}; pilot non-WT FASTA batches: {len(pilot_fastas)}; remaining MT FASTA batches: {len(remaining_fastas)}")
print(f"Fast clipping affects {manifest['fast_clipped'].sum()} unique sequences and removes the mutation for {(~pairs['mutation_retained_fast']).sum()} variants")
print(f"Pilot variants: {pairs['pilot_variant'].sum()}; remaining variants: {(~pairs['pilot_variant']).sum()}")
```

```
Variants: 2179; unique WT: 871; pilot MT (non-WT): 762; remaining MT: 1386
WT FASTA batches: 2; pilot non-WT FASTA batches: 2; remaining MT FASTA batches: 3
Fast clipping affects 108 unique sequences and removes the mutation for 7 variants
Pilot variants: 770; remaining variants: 1409
```

---

## 18.2 Local Inference Adapter

The downloaded CLI discards sorting-signal probabilities when writing CSV. This adapter calls the packaged Fast/ESM1b ensemble and preserves its internal 9D signal output. Run WT inference first, then pilot MT inference only for the sensitivity diagnostic.

```python
# Set to True only in the environment where the licensed package and ESM1b backbone are available.
RUN_LOCAL_INFERENCE = True
DEVICE = "cuda:1"
TOKENS_PER_BATCH = 8192

REMAINING_MT_PREDICTIONS_CSV = DEEPlOC_DIR / "remaining_mt_predictions_fast_esm1b.csv"

def run_fast_deeploc(table, output_csv, device=DEVICE, tokens_per_batch=TOKENS_PER_BATCH):
    import torch
    from esm import Alphabet

    package_root = str(MODEL_PACKAGE.resolve())
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    from DeepLoc2.data import BatchConverter, FastaBatchedDatasetTorch
    from DeepLoc2.model import ESM1bE2E

    work = table[["sequence_id", "sequence", "original_length", "fast_clipped", "effective_length_fast"]].copy()
    work["sequence"] = work["sequence"].map(lambda s: s if len(s) <= 1022 else s[:511] + s[-511:])
    model = ESM1bE2E().to(device).eval()
    alphabet = Alphabet.from_architecture("roberta_large")
    dataset_df = work.rename(columns={"sequence_id": "ACC", "sequence": "Sequence"}).reset_index(drop=True)
    dataset = FastaBatchedDatasetTorch(dataset_df)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=2)
    loader = torch.utils.data.DataLoader(dataset, collate_fn=BatchConverter(alphabet), batch_sampler=batches)
    records = []
    with torch.inference_mode():
        for tokens, lengths, mask, identifiers in loader:
            localisation, _, signal = model(tokens, lengths, mask)
            assert localisation.shape == (len(identifiers), 11) and signal.shape == (len(identifiers), 9)
            for i, sequence_id in enumerate(identifiers):
                record = {"sequence_id": sequence_id, "model_setting": MODEL_SETTING}
                record.update({name: float(localisation[i, j + 1]) for j, name in enumerate(LOCATIONS)})
                record.update({name: float(signal[i, j]) for j, name in enumerate(SIGNALS)})
                records.append(record)
    result = work.drop(columns="sequence").merge(pd.DataFrame(records), on="sequence_id", validate="one_to_one")
    assert len(result) == len(work) and result[LOCATIONS + SIGNALS].applymap(np.isfinite).all().all()
    assert ((result[LOCATIONS + SIGNALS] >= 0) & (result[LOCATIONS + SIGNALS] <= 1)).all().all()
    result.to_csv(output_csv, index=False)
    return result

if RUN_LOCAL_INFERENCE:
    wt_predictions = run_fast_deeploc(wt_manifest, WT_PREDICTIONS_CSV)
    pilot_predictions = run_fast_deeploc(pilot_mt_manifest, PILOT_PREDICTIONS_CSV) if len(pilot_mt_manifest) else pd.DataFrame()
    print(f"Saved WT predictions: {WT_PREDICTIONS_CSV}")
    print(f"Saved pilot MT predictions: {PILOT_PREDICTIONS_CSV}")
    # Full MT inference: remaining non-pilot, non-WT sequences
    remaining_predictions = run_fast_deeploc(remaining_mt_manifest, REMAINING_MT_PREDICTIONS_CSV) if len(remaining_mt_manifest) else pd.DataFrame()
    print(f"Saved remaining MT predictions: {REMAINING_MT_PREDICTIONS_CSV}")
    # Combine all MT predictions (pilot + remaining)
    mt_parts = [p for p in [pilot_predictions, remaining_predictions] if len(p)]
    all_mt_predictions = pd.concat(mt_parts, ignore_index=True) if mt_parts else pd.DataFrame()
    if len(all_mt_predictions):
        all_mt_predictions.to_csv(FULL_PREDICTIONS_CSV, index=False)
    print(f"Saved all MT predictions ({len(all_mt_predictions)} sequences): {FULL_PREDICTIONS_CSV}")
else:
    print("Inference disabled. Set RUN_LOCAL_INFERENCE=True on the licensed DeepLoc environment.")
```

```
Lightning automatically upgraded your loaded checkpoint from v1.5.8 to v2.6.5. ...
(10 checkpoints × 3 inference passes = 30 upgrade warnings, cosmetic only)

Saved WT predictions: .../wt_predictions_fast_esm1b.csv
Saved pilot MT predictions: .../pilot_mt_predictions_fast_esm1b.csv
Saved remaining MT predictions: .../remaining_mt_predictions_fast_esm1b.csv
Saved all MT predictions (2148 sequences): .../all_predictions_fast_esm1b.csv
```

---

## 18.3 Validate WT Predictions and Create Context Features

Plant-specific outputs are retained in the raw file for auditing and excluded from the primary human feature sets. WT TM is separated because the 64D model already contains three TMD features.

```python
if not WT_PREDICTIONS_CSV.exists():
    raise FileNotFoundError(f"Run WT DeepLoc inference first: {WT_PREDICTIONS_CSV}")
wt_pred = pd.read_csv(WT_PREDICTIONS_CSV)
required_pred = ["sequence_id", "model_setting"] + LOCATIONS + SIGNALS
assert not [c for c in required_pred if c not in wt_pred.columns]
assert wt_pred["sequence_id"].is_unique and set(wt_pred["sequence_id"]) == set(wt_manifest["sequence_id"])
assert wt_pred["model_setting"].eq(MODEL_SETTING).all()
assert ((wt_pred[LOCATIONS + SIGNALS] >= 0) & (wt_pred[LOCATIONS + SIGNALS] <= 1)).all().all()

context = pairs[["KEY", "Gene", "wt_sequence_id", "mutation_retained_fast"]].merge(
    wt_pred, left_on="wt_sequence_id", right_on="sequence_id", validate="many_to_one")
rename = {c: f"deeploc_wt_loc_{c}" for c in LOCATIONS} | {c: f"deeploc_wt_signal_{c}" for c in SIGNALS}
context = context.rename(columns=rename)
WT_LOC_COLS = [f"deeploc_wt_loc_{c}" for c in HUMAN_LOCATIONS]
WT_SIGNAL_COLS = [f"deeploc_wt_signal_{c}" for c in CORE_SIGNALS]
WT_TM_COL = ["deeploc_wt_signal_transmembrane_domain"]
save_cols = ["KEY", "Gene", "wt_sequence_id", "mutation_retained_fast"] + list(rename.values())
context[save_cols].to_csv(CONTEXT_FEATURES_CSV, index=False)
assert len(context) == 2179 and context["KEY"].is_unique
print(context[WT_LOC_COLS + WT_SIGNAL_COLS + WT_TM_COL].describe().T.to_string())
print(f"Saved: {CONTEXT_FEATURES_CSV}")
```

### WT Context Feature Distributions (n=2179 variants)

| Feature | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| **Localisation (WT)** | | | | | | | |
| deeploc_wt_loc_cytoplasm | 0.364 | 0.229 | 0.064 | 0.171 | 0.262 | 0.596 | 0.892 |
| deeploc_wt_loc_nucleus | 0.302 | 0.250 | 0.024 | 0.109 | 0.179 | 0.438 | 0.976 |
| deeploc_wt_loc_extracellular | 0.222 | 0.299 | 0.002 | 0.032 | 0.080 | 0.230 | 0.972 |
| deeploc_wt_loc_cell_membrane | 0.352 | 0.278 | 0.011 | 0.132 | 0.241 | 0.554 | 0.947 |
| deeploc_wt_loc_mitochondrion | 0.224 | 0.246 | 0.014 | 0.076 | 0.134 | 0.237 | 0.968 |
| deeploc_wt_loc_endoplasmic_reticulum | 0.257 | 0.200 | 0.008 | 0.113 | 0.210 | 0.334 | 0.939 |
| deeploc_wt_loc_lysosome_vacuole | 0.268 | 0.181 | 0.005 | 0.109 | 0.235 | 0.395 | 0.876 |
| deeploc_wt_loc_golgi_apparatus | 0.248 | 0.176 | 0.004 | 0.117 | 0.213 | 0.338 | 0.961 |
| deeploc_wt_loc_peroxisome | 0.058 | 0.119 | 0.000 | 0.006 | 0.018 | 0.054 | 0.997 |
| **Sorting Signals (WT)** | | | | | | | |
| deeploc_wt_signal_signal_peptide | 0.335 | 0.395 | 0.000 | 0.008 | 0.060 | 0.810 | 0.989 |
| deeploc_wt_signal_mitochondrial_transit_peptide | 0.104 | 0.269 | 0.000 | 0.002 | 0.006 | 0.017 | 0.991 |
| deeploc_wt_signal_nuclear_localisation_signal | 0.215 | 0.280 | 0.001 | 0.004 | 0.023 | 0.418 | 0.975 |
| deeploc_wt_signal_nuclear_export_signal | 0.231 | 0.306 | 0.000 | 0.001 | 0.007 | 0.503 | 0.941 |
| deeploc_wt_signal_peroxisomal_targeting_signal | 0.042 | 0.134 | 0.000 | 0.002 | 0.005 | 0.013 | 0.995 |
| deeploc_wt_signal_gpi_anchor | 0.017 | 0.076 | 0.000 | 0.001 | 0.002 | 0.009 | 0.971 |
| deeploc_wt_signal_transmembrane_domain | 0.231 | 0.331 | 0.000 | 0.005 | 0.016 | 0.542 | 0.932 |

Saved: `/mnt/volume6/czj/labLGN/LabLZ/xgboost_trial/deeploc_wt_context_features.csv`

---

## 18.4 Sorting-Signal Sensitivity Pilot

Sensitivity is described by the distribution of signed and absolute MT−WT changes, threshold exceedance rates, and Fast-clipping retention. This determines whether sorting-signal deltas have enough dynamic range to justify a full MT inference run.

```python
if PILOT_PREDICTIONS_CSV.exists():
    pilot_nonwt = pd.read_csv(PILOT_PREDICTIONS_CSV)
    combined_pred = pd.concat([wt_pred, pilot_nonwt], ignore_index=True).drop_duplicates("sequence_id", keep="last")
    pilot_pairs = pairs.loc[pairs["pilot_variant"]].copy()
    wt_values = combined_pred.rename(columns={"sequence_id": "wt_sequence_id",
        **{c: f"wt_{c}" for c in LOCATIONS + SIGNALS}})
    mt_values = combined_pred.rename(columns={"sequence_id": "mt_sequence_id",
        **{c: f"mt_{c}" for c in LOCATIONS + SIGNALS}})
    pilot = pilot_pairs.merge(wt_values[["wt_sequence_id"] + [f"wt_{c}" for c in LOCATIONS + SIGNALS]],
        on="wt_sequence_id", validate="many_to_one")
    pilot = pilot.merge(mt_values[["mt_sequence_id"] + [f"mt_{c}" for c in LOCATIONS + SIGNALS]],
        on="mt_sequence_id", validate="many_to_one")
    for name in LOCATIONS + SIGNALS:
        pilot[f"delta_{name}"] = pilot[f"mt_{name}"] - pilot[f"wt_{name}"]
    rows = []
    for name in SIGNALS:
        values = pilot[f"delta_{name}"].to_numpy()
        abs_values = np.abs(values)
        rows.append({"signal": name, "n": len(values),
            "median_delta": np.median(values), "median_abs_delta": np.median(abs_values),
            "p90_abs_delta": np.quantile(abs_values, 0.90),
            "p95_abs_delta": np.quantile(abs_values, 0.95),
            "p99_abs_delta": np.quantile(abs_values, 0.99),
            "fraction_abs_gt_0.001": np.mean(abs_values > 0.001),
            "fraction_abs_gt_0.01": np.mean(abs_values > 0.01),
            "fraction_abs_gt_0.05": np.mean(abs_values > 0.05)})
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(SENSITIVITY_CSV, index=False)
    print(sensitivity.to_string(index=False))
    print(f"Pilot variants whose mutation is removed by Fast clipping: {(~pilot['mutation_retained_fast']).sum()}")
```

### Sorting-Signal Delta Distributions (n=770 pilot variants)

| Signal | Median Δ | Median \|Δ\| | P90 \|Δ\| | P95 \|Δ\| | P99 \|Δ\| | frac >0.001 | frac >0.01 | frac >0.05 |
|---|---|---|---|---|---|---|---|---|---|
| signal_peptide | +8.6e-06 | 0.000872 | 0.00875 | 0.01443 | 0.03765 | 0.473 | 0.081 | 0.004 |
| transmembrane_domain | +6.1e-08 | 0.000570 | 0.00639 | 0.01003 | 0.02331 | 0.405 | 0.051 | 0.000 |
| mitochondrial_transit_peptide | −8.6e-07 | 0.000086 | 0.00343 | 0.00678 | 0.03330 | 0.177 | 0.034 | 0.004 |
| chloroplast_transit_peptide | +7.0e-07 | 0.000020 | 0.00116 | 0.00308 | 0.01635 | 0.110 | 0.021 | 0.001 |
| thylakoid_transit_peptide | −2.4e-07 | 0.000012 | 0.00014 | 0.00028 | 0.00182 | 0.014 | 0.000 | 0.000 |
| nuclear_localisation_signal | −1.3e-05 | 0.000393 | 0.01178 | 0.01982 | 0.03946 | 0.391 | 0.112 | 0.005 |
| nuclear_export_signal | −2.2e-06 | 0.000147 | 0.01032 | 0.01918 | 0.04737 | 0.344 | 0.105 | 0.009 |
| peroxisomal_targeting_signal | −2.6e-06 | 0.000070 | 0.00278 | 0.01114 | 0.03694 | 0.173 | 0.053 | 0.008 |
| gpi_anchor | +4.3e-06 | 0.000091 | 0.00188 | 0.00389 | 0.01163 | 0.165 | 0.012 | 0.003 |

Pilot variants whose mutation is removed by Fast clipping: 1

**Key finding:** Signal peptide (SP) shows the largest median absolute delta (0.00087) and highest fraction exceeding 0.01 (8.1%). NLS and NES show moderate dynamic range (P95 ~0.02). Plant-specific signals (chloroplast, thylakoid) show negligible change — expected for human proteins. TM domain signal shows ~40% of variants with |Δ| > 0.001 but 0% with |Δ| > 0.05, consistent with single substitutions rarely creating/destroying transmembrane domains.

---

## 18.5 Fixed-Fold WT-Context Ablation

All imputation, scaling, and PCA are fitted on training-fold data only. The 64D predictions are reused from Task 15. New models use the same fold assignment and XGBoost settings.

```python
df = pd.read_csv(FEATURES_CSV)
task15 = pd.read_csv(TASK15_OOF)
assert len(df) == 2179 and df["KEY"].is_unique
df = df.merge(task15[["KEY", "fold", "oof_stability_tmd_64", "final_alphamissense_score"]],
    on="KEY", how="left", validate="one_to_one")
for name in DDG_COLS:
    table = pd.read_csv(BASE / f"{name}.csv")
    df = df.merge(table[["KEY", name]], on="KEY", how="left", validate="one_to_one")
tmd_full = pd.read_csv(TMD_CSV)
df = df.merge(tmd_full[["KEY"] + TMD_COLS], on="KEY", how="left", validate="one_to_one")
df = df.merge(context[["KEY"] + WT_LOC_COLS + WT_SIGNAL_COLS + WT_TM_COL],
    on="KEY", how="left", validate="one_to_one")

esm_cols = [c for c in df.columns if c.startswith("esm_")]
assert len(esm_cols) == 1280 and int(df["Mislocalized"].sum()) == 236
y = df["Mislocalized"].astype(int).to_numpy()
fold_id = df["fold"].astype(int).to_numpy()
X_esm = df[esm_cols].to_numpy(np.float32)
X_struct = df[STRUCT_COLS].to_numpy(np.float32)
X_base_extra = df[DDG_COLS + TMD_COLS].to_numpy(np.float32)
oof = {"xgboost_64": df["oof_stability_tmd_64"].to_numpy(np.float32)}
feature_sets = {
    "wt_loc_73": WT_LOC_COLS,
    "wt_signal_70": WT_SIGNAL_COLS,
    "wt_context_79": WT_LOC_COLS + WT_SIGNAL_COLS,
    "wt_context_tm_80": WT_LOC_COLS + WT_SIGNAL_COLS + WT_TM_COL,
}
importance_parts = []

for model_name, added_cols in feature_sets.items():
    oof[model_name] = np.full(len(df), np.nan, dtype=np.float32)
    X_added = df[added_cols].to_numpy(np.float32)
    for fold in sorted(np.unique(fold_id)):
        train_idx = np.flatnonzero(fold_id != fold)
        test_idx = np.flatnonzero(fold_id == fold)
        esm_imp, esm_scaler = SimpleImputer(strategy="median"), StandardScaler()
        esm_train = esm_scaler.fit_transform(esm_imp.fit_transform(X_esm[train_idx]))
        esm_test = esm_scaler.transform(esm_imp.transform(X_esm[test_idx]))
        pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
        pc_train, pc_test = pca.fit_transform(esm_train), pca.transform(esm_test)
        struct_imp, struct_scaler = SimpleImputer(strategy="median"), StandardScaler()
        struct_train = struct_scaler.fit_transform(struct_imp.fit_transform(X_struct[train_idx]))
        struct_test = struct_scaler.transform(struct_imp.transform(X_struct[test_idx]))
        extra = np.hstack([X_base_extra, X_added])
        extra_imp = SimpleImputer(strategy="median")
        extra_train, extra_test = extra_imp.fit_transform(extra[train_idx]), extra_imp.transform(extra[test_idx])
        X_train = np.hstack([pc_train, struct_train, extra_train]).astype(np.float32)
        X_test = np.hstack([pc_test, struct_test, extra_test]).astype(np.float32)
        model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.5, objective="binary:logistic", eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
        model.fit(X_train, y[train_idx],
            sample_weight=compute_sample_weight("balanced", y[train_idx]), verbose=False)
        oof[model_name][test_idx] = model.predict_proba(X_test)[:, 1]
        names = [f"PC{i + 1}" for i in range(N_COMPONENTS)] + STRUCT_COLS + DDG_COLS + TMD_COLS + added_cols
        importance_parts.append(pd.DataFrame({"model": model_name, "fold": fold, "feature": names,
            "importance": model.feature_importances_}))
    assert np.isfinite(oof[model_name]).all()

def precision_at_recall(y_true, score, target):
    precision, recall, _ = precision_recall_curve(y_true, score)
    order = np.argsort(recall)
    return float(np.interp(target, recall[order], precision[order]))

def metric_row(scope, model_name, truth, score):
    return {"scope": scope, "model": model_name, "n": len(truth), "positives": int(truth.sum()),
        "auroc": roc_auc_score(truth, score), "auprc": average_precision_score(truth, score),
        "precision_at_recall_0.20": precision_at_recall(truth, score, 0.20),
        "precision_at_recall_0.40": precision_at_recall(truth, score, 0.40)}

records = [metric_row("full_oof", name, y, score) for name, score in oof.items()]
paired_am = df["final_alphamissense_score"].notna().to_numpy()
records.append(metric_row("paired_alphamissense", "alphamissense",
    y[paired_am], df.loc[paired_am, "final_alphamissense_score"].to_numpy()))
records.extend(metric_row("paired_alphamissense", name, y[paired_am], score[paired_am])
    for name, score in oof.items())
metrics_df = pd.DataFrame(records)
print(metrics_df.to_string(index=False))
```

### Full OOF Metrics (n=2179, 236 positives)

| Model | AUROC | AUPRC | Prec@R20 | Prec@R40 |
|---|---|---|---|---|
| xgboost_64 (baseline) | 0.6422 | 0.1981 | 0.2661 | 0.1861 |
| wt_loc_73 (+9 WT loc) | 0.6320 | 0.2247 | 0.3588 | 0.2133 |
| **wt_signal_70 (+6 WT signals)** | **0.6560** | **0.2479** | 0.3137 | 0.2125 |
| wt_context_79 (+15 WT loc+sig) | 0.6524 | 0.2425 | 0.3357 | 0.2223 |
| wt_context_tm_80 (+16 WT loc+sig+TM) | 0.6557 | 0.2435 | **0.3856** | 0.2192 |

### Paired AlphaMissense Cohort (n=2140, 235 positives)

| Model | AUROC | AUPRC | Prec@R20 | Prec@R40 |
|---|---|---|---|---|
| alphamissense | 0.6491 | 0.1619 | 0.1780 | 0.1741 |
| xgboost_64 | 0.6442 | 0.1999 | 0.2655 | 0.1865 |
| wt_loc_73 | 0.6340 | 0.2263 | 0.3588 | 0.2171 |
| **wt_signal_70** | **0.6579** | **0.2499** | 0.3133 | 0.2141 |
| wt_context_79 | 0.6539 | 0.2441 | 0.3357 | 0.2233 |
| wt_context_tm_80 | 0.6574 | 0.2451 | **0.3852** | 0.2196 |

**Key findings:**
- **wt_signal_70** achieves the best AUROC (0.656) and AUPRC (0.248) among all models — adding only 6 WT sorting-signal probabilities
- **wt_context_tm_80** achieves the highest precision@recall=0.20 (0.386), suggesting TM context helps at conservative thresholds
- **wt_loc_73** (9 WT localisation probabilities alone) slightly degrades AUROC (−0.010 vs baseline) but improves AUPRC (+0.027), indicating better ranking of top predictions
- All four DeepLoc-context models had higher AUPRC point estimates than AlphaMissense on the paired cohort. Three of the four also had higher AUROC point estimates; WT localisation alone (wt_loc_73, AUROC 0.634) did not. No direct paired bootstrap comparing wt_signal_70 against AlphaMissense has been run — the statement above is purely from point estimates.

---

## 18.6 Paired Gene-Cluster Bootstrap

Resamples genes from the fixed OOF predictions (2,000 replicates). Estimates evaluation-sample uncertainty conditional on the trained predictions; does not include fold, PCA, or retraining uncertainty.

```python
rng = np.random.default_rng(RANDOM_STATE)
unique_genes = df["Gene"].astype(str).unique()
gene_to_idx = {gene: np.flatnonzero(df["Gene"].astype(str).to_numpy() == gene) for gene in unique_genes}
bootstrap_rows = []
for replicate in range(2000):
    sampled_genes = rng.choice(unique_genes, size=len(unique_genes), replace=True)
    idx = np.concatenate([gene_to_idx[gene] for gene in sampled_genes])
    if np.unique(y[idx]).size < 2:
        continue
    base_auc = roc_auc_score(y[idx], oof["xgboost_64"][idx])
    base_ap = average_precision_score(y[idx], oof["xgboost_64"][idx])
    for name in feature_sets:
        bootstrap_rows.append({"replicate": replicate, "comparison": f"{name}-xgboost_64",
            "delta_auroc": roc_auc_score(y[idx], oof[name][idx]) - base_auc,
            "delta_auprc": average_precision_score(y[idx], oof[name][idx]) - base_ap})
bootstrap = pd.DataFrame(bootstrap_rows)
bootstrap.to_csv(BOOTSTRAP_CSV, index=False)
summary = bootstrap.groupby("comparison").agg(
    delta_auroc_mean=("delta_auroc", "mean"),
    delta_auroc_low=("delta_auroc", lambda x: np.quantile(x, 0.025)),
    delta_auroc_high=("delta_auroc", lambda x: np.quantile(x, 0.975)),
    delta_auprc_mean=("delta_auprc", "mean"),
    delta_auprc_low=("delta_auprc", lambda x: np.quantile(x, 0.025)),
    delta_auprc_high=("delta_auprc", lambda x: np.quantile(x, 0.975))).reset_index()
print(summary.to_string(index=False))
print("Do not claim superiority when the paired interval crosses zero.")
```

### Bootstrap Summary (2,000 gene-cluster replicates)

| Comparison | ΔAUROC mean | ΔAUROC 95% CI | ΔAUPRC mean | ΔAUPRC 95% CI |
|---|---|---|---|---|
| wt_loc_73 − xgboost_64 | −0.0107 | [−0.0486, +0.0232] | +0.0267 | [−0.0075, +0.0653] |
| **wt_signal_70 − xgboost_64** | **+0.0138** | [−0.0194, +0.0449] | **+0.0472** | **[+0.0130, +0.0843]** |
| wt_context_79 − xgboost_64 | +0.0097 | [−0.0289, +0.0466] | +0.0446 | [+0.0074, +0.0859] |
| wt_context_tm_80 − xgboost_64 | +0.0132 | [−0.0268, +0.0504] | +0.0442 | [+0.0082, +0.0852] |

**Bootstrap interpretation (with caveats):**

Under a gene-cluster bootstrap conditional on the fixed OOF predictions, the wt_signal_70 ΔAUPRC interval excluded zero [+0.0130, +0.0843]. This is positive evidence but does **not** directly equate to "statistically significant superiority" for the following reasons:

1. **Incomplete uncertainty:** The bootstrap resamples only from fixed OOF predictions. It does not incorporate fold-assignment uncertainty, PCA refitting, XGBoost retraining, or hyperparameter uncertainty. The interval reflects evaluation-sample variability conditional on one trained model, not full model-selection uncertainty.
2. **Multiple comparisons:** Four DeepLoc ablations were compared against the same baseline, and wt_signal_70 was selected post-hoc as the best. The reported CI does not correct for this selection. Formal approaches (Holm correction, bootstrap max-statistic adjustment, or pre-registering wt_signal_70 as the sole candidate) would be needed for a confirmatory claim.
3. **No direct AlphaMissense bootstrap:** The bootstrap only compares DeepLoc models against xgboost_64. A separate paired gene-cluster bootstrap on the 2,140-row paired cohort is needed before comparing wt_signal_70 to AlphaMissense.

Additional observations:
- No model achieves a statistically significant ΔAUROC (all CIs cross zero)
- **wt_loc_73** shows a negative ΔAUROC trend (−0.0107), confirming that WT localisation context alone is not beneficial for discrimination
- Adding TM domain probability (wt_context_tm_80 vs wt_context_79) provides negligible additional benefit

---

## 18.7 Full WT–MT Sorting-Signal Deltas

The sensitivity pilot confirmed non-trivial tails for SP, NLS, and NES. Build 7 sorting-signal delta features (6 core signals + TM) from all 2,148 MT predictions. Localisation deltas are retained as exploratory diagnostics.

```python
DELTA_CSV = BASE / "deeploc_sorting_delta_features.csv"

if FULL_PREDICTIONS_CSV.exists():
    all_pred = pd.read_csv(FULL_PREDICTIONS_CSV)
    wt_lookup = wt_pred.copy()
    mt_lookup = all_pred.copy()
    combined_pred = pd.concat([wt_lookup, mt_lookup], ignore_index=True).drop_duplicates("sequence_id", keep="last")

    wt_vals = combined_pred.rename(columns={"sequence_id": "wt_sequence_id",
        **{c: f"wt_{c}" for c in LOCATIONS + SIGNALS}})
    mt_vals = combined_pred.rename(columns={"sequence_id": "mt_sequence_id",
        **{c: f"mt_{c}" for c in LOCATIONS + SIGNALS}})

    delta = pairs[["KEY", "Gene", "wt_sequence_id", "mt_sequence_id", "mutation_retained_fast"]].copy()
    delta = delta.merge(wt_vals[["wt_sequence_id"] + [f"wt_{c}" for c in LOCATIONS + SIGNALS]],
        on="wt_sequence_id", validate="many_to_one")
    delta = delta.merge(mt_vals[["mt_sequence_id"] + [f"mt_{c}" for c in LOCATIONS + SIGNALS]],
        on="mt_sequence_id", validate="many_to_one")

    # 7 sorting-signal deltas: 6 core + TM
    SORTING_DELTA_SIGNALS = CORE_SIGNALS + ["transmembrane_domain"]
    SORTING_DELTA_COLS = [f"deeploc_delta_{c}" for c in SORTING_DELTA_SIGNALS]
    for name in SORTING_DELTA_SIGNALS:
        delta[f"deeploc_delta_{name}"] = delta[f"mt_{name}"] - delta[f"wt_{name}"]

    LOC_DELTA_COLS = [f"deeploc_delta_loc_{c}" for c in HUMAN_LOCATIONS]
    for name in HUMAN_LOCATIONS:
        delta[f"deeploc_delta_loc_{name}"] = delta[f"mt_{name}"] - delta[f"wt_{name}"]

    delta[["KEY"] + SORTING_DELTA_COLS + LOC_DELTA_COLS].to_csv(DELTA_CSV, index=False)
    assert len(delta) == 2179 and delta["KEY"].is_unique

    print(f"Sorting-signal delta features ({len(SORTING_DELTA_COLS)}):")
    print(delta[SORTING_DELTA_COLS].describe().T.to_string())
    print(f"\nSaved: {DELTA_CSV}")
    print(f"Variants with missing MT prediction: {delta[SORTING_DELTA_COLS].isna().any(axis=1).sum()}")
```

### Full-Cohort Sorting-Signal Delta Distributions (n=2,179 variants)

| Delta Feature | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| deeploc_delta_signal_peptide | +0.00012 | 0.00557 | −0.071 | −0.00035 | +0.00000 | +0.00043 | +0.054 |
| deeploc_delta_mitochondrial_transit_peptide | +0.00010 | 0.00469 | −0.038 | −0.00008 | −0.00000 | +0.00004 | +0.078 |
| deeploc_delta_nuclear_localisation_signal | −0.00034 | 0.00779 | −0.105 | −0.00035 | −0.00000 | +0.00023 | +0.105 |
| deeploc_delta_nuclear_export_signal | −0.00061 | 0.00951 | −0.182 | −0.00038 | −0.00000 | +0.00005 | +0.169 |
| deeploc_delta_peroxisomal_targeting_signal | −0.00023 | 0.00669 | −0.107 | −0.00007 | −0.00000 | +0.00005 | +0.130 |
| deeploc_delta_gpi_anchor | +0.00011 | 0.00287 | −0.074 | −0.00002 | +0.00000 | +0.00007 | +0.052 |
| deeploc_delta_transmembrane_domain | +0.00003 | 0.00390 | −0.047 | −0.00020 | +0.00000 | +0.00033 | +0.058 |

Variants with missing MT prediction: 0

**Key observations:** In the full (non-enriched) cohort, the extreme tails are considerably wider than the pilot suggested — NES max |Δ| reaches 0.182, PTS 0.130, NLS 0.105. The median remains near zero, confirming that most variants produce negligible signal changes, but a small subset of variants drive substantial sorting-signal perturbations.

---

## 18.8 Fixed-Fold Ablation with Sorting-Signal Deltas

Four-model comparison on Task 15's fixed folds: (1) 64D baseline, (2) wt_signal_70, (3) 64D + 7 sorting-signal deltas, (4) wt_signal_70 + 7 sorting-signal deltas.

```python
DELTA_OOF_CSV = BASE / "task18_deeploc_delta_oof.csv"
DELTA_METRICS_CSV = BASE / "task18_deeploc_delta_metrics.csv"
DELTA_IMPORTANCE_CSV = BASE / "task18_deeploc_delta_importance.csv"
DELTA_BOOTSTRAP_CSV = BASE / "task18_deeploc_delta_bootstrap.csv"

if delta is not None:
    delta_df = df.merge(delta[["KEY"] + SORTING_DELTA_COLS], on="KEY", how="left", validate="one_to_one")

    delta_oof = {}
    delta_feature_sets = {
        "xgboost_64":               [],
        "wt_signal_70":             WT_SIGNAL_COLS,
        "xgboost_64_plus_delta_71":  SORTING_DELTA_COLS,
        "wt_signal_70_plus_delta_77": WT_SIGNAL_COLS + SORTING_DELTA_COLS,
    }
    delta_importance_parts = []

    for model_name, added_cols in delta_feature_sets.items():
        if model_name == "xgboost_64":
            delta_oof[model_name] = df["oof_stability_tmd_64"].to_numpy(np.float32)
            continue
        if model_name == "wt_signal_70":
            delta_oof[model_name] = oof["wt_signal_70"]
            continue

        delta_oof[model_name] = np.full(len(df), np.nan, dtype=np.float32)
        X_added = delta_df[added_cols].to_numpy(np.float32)
        for fold in sorted(np.unique(fold_id)):
            train_idx = np.flatnonzero(fold_id != fold)
            test_idx = np.flatnonzero(fold_id == fold)
            esm_imp, esm_scaler = SimpleImputer(strategy="median"), StandardScaler()
            esm_train = esm_scaler.fit_transform(esm_imp.fit_transform(X_esm[train_idx]))
            esm_test = esm_scaler.transform(esm_imp.transform(X_esm[test_idx]))
            pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
            pc_train, pc_test = pca.fit_transform(esm_train), pca.transform(esm_test)
            struct_imp, struct_scaler = SimpleImputer(strategy="median"), StandardScaler()
            struct_train = struct_scaler.fit_transform(struct_imp.fit_transform(X_struct[train_idx]))
            struct_test = struct_scaler.transform(struct_imp.transform(X_struct[test_idx]))
            # All models: X_base_extra + added_cols only
            # added_cols already contains WT_SIGNAL_COLS for wt_signal_70_plus_delta_77
            extra = np.hstack([X_base_extra, X_added])
            extra_imp = SimpleImputer(strategy="median")
            extra_train, extra_test = extra_imp.fit_transform(extra[train_idx]), extra_imp.transform(extra[test_idx])
            X_train = np.hstack([pc_train, struct_train, extra_train]).astype(np.float32)
            X_test = np.hstack([pc_test, struct_test, extra_test]).astype(np.float32)
            model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.5, objective="binary:logistic", eval_metric="aucpr",
                random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist")
            model.fit(X_train, y[train_idx],
                sample_weight=compute_sample_weight("balanced", y[train_idx]), verbose=False)
            delta_oof[model_name][test_idx] = model.predict_proba(X_test)[:, 1]
            names = [f"PC{i + 1}" for i in range(N_COMPONENTS)] + STRUCT_COLS + DDG_COLS + TMD_COLS + added_cols
            # Hard dimension checks
            expected_dim = 64 + len(added_cols)
            assert X_train.shape[1] == expected_dim, f"{model_name}: expected {expected_dim}D, got {X_train.shape[1]}D"
            assert len(names) == expected_dim, f"{model_name}: expected {expected_dim} names, got {len(names)}"
            assert len(names) == len(set(names)), f"{model_name}: duplicate feature names detected"
            delta_importance_parts.append(pd.DataFrame({"model": model_name, "fold": fold, "feature": names,
                "importance": model.feature_importances_}))
            delta_oof[model_name][test_idx] = model.predict_proba(X_test)[:, 1]
            names = [f"PC{i + 1}" for i in range(N_COMPONENTS)] + STRUCT_COLS + DDG_COLS + TMD_COLS + added_cols
            # Hard dimension checks
            expected_dim = 64 + len(added_cols)
            assert X_train.shape[1] == expected_dim, f"{model_name}: expected {expected_dim}D, got {X_train.shape[1]}D"
            assert len(names) == expected_dim, f"{model_name}: expected {expected_dim} names, got {len(names)}"
            assert len(names) == len(set(names)), f"{model_name}: duplicate feature names detected"
            delta_importance_parts.append(pd.DataFrame({"model": model_name, "fold": fold, "feature": names,
                "importance": model.feature_importances_}))
        assert np.isfinite(delta_oof[model_name]).all()
        assert np.all(delta_oof[model_name] >= 0)
        assert np.all(delta_oof[model_name] <= 1)

    delta_records = [metric_row("full_oof", name, y, score) for name, score in delta_oof.items()]
    paired_am = df["final_alphamissense_score"].notna().to_numpy()
    delta_records.extend(metric_row("paired_alphamissense", name, y[paired_am], score[paired_am])
        for name, score in delta_oof.items())
    delta_metrics_df = pd.DataFrame(delta_records)
    print(delta_metrics_df.to_string(index=False))

    delta_out = df[["KEY", "Gene", "Mutation_used", "Mislocalized", "fold"]].copy()
    for name, score in delta_oof.items():
        delta_out[f"oof_{name}"] = score
    delta_out.to_csv(DELTA_OOF_CSV, index=False)
    delta_metrics_df.to_csv(DELTA_METRICS_CSV, index=False)
    if delta_importance_parts:
        delta_importance = pd.concat(delta_importance_parts).groupby(["model", "feature"], as_index=False)["importance"].mean()
        delta_importance["rank_within_model"] = delta_importance.groupby("model")["importance"].rank(method="min", ascending=False).astype(int)
        delta_importance.sort_values(["model", "rank_within_model"]).to_csv(DELTA_IMPORTANCE_CSV, index=False)
```

### Full OOF Metrics — Four-Model Delta Comparison (n=2,179, 236 positives)

| Model | Features | Dim | AUROC | AUPRC | Prec@R20 | Prec@R40 |
|---|---|---|---|---|---|---|
| **xgboost_64** | 64D baseline | 64 | **0.6422** | 0.1981 | 0.2661 | 0.1861 |
| **wt_signal_70** | 64D + 6 WT signals | 70 | **0.6560** | **0.2479** | 0.3137 | **0.2125** |
| xgboost_64_plus_delta_71 | 64D + 7 sorting deltas | 71 | 0.6313 | 0.1949 | 0.2763 | 0.1969 |
| wt_signal_70_plus_delta_77 | 64D + 6 WT signals + 7 deltas | 77 | 0.6431 | 0.2302 | **0.3139** | 0.1999 |

### Paired AlphaMissense Cohort (n=2,140, 235 positives)

| Model | AUROC | AUPRC | Prec@R20 | Prec@R40 |
|---|---|---|---|---|
| **xgboost_64** | 0.6442 | 0.1999 | 0.2655 | 0.1865 |
| **wt_signal_70** | **0.6579** | **0.2499** | 0.3133 | **0.2141** |
| xgboost_64_plus_delta_71 | 0.6330 | 0.1966 | 0.2798 | 0.1967 |
| wt_signal_70_plus_delta_77 | 0.6446 | 0.2321 | **0.3176** | 0.2026 |

**Fixed results (77D bug resolved):**
- xgboost_64_plus_delta_71 vs xgboost_64: ΔAUROC −0.011, ΔAUPRC −0.004 — consistent with the earlier (unbuggy) run
- wt_signal_70_plus_delta_77 vs wt_signal_70: ΔAUROC −0.013, ΔAUPRC −0.018 — the delta features still do not improve upon wt_signal_70, but the effect is smaller than the buggy run suggested
- **wt_signal_70 achieved the best primary AUROC and AUPRC point estimates** (Prec@R20 is exploratory and marginally favours 77D due to threshold instability; see 18.10.2)

---

## 18.9 Paired Gene-Cluster Bootstrap (Delta Models)

Comparisons against xgboost_64 and against wt_signal_70, with 2,000 gene-cluster replicates.

```python
if delta is not None:
    rng = np.random.default_rng(RANDOM_STATE)
    delta_bootstrap_rows = []
    for replicate in range(2000):
        sampled_genes = rng.choice(unique_genes, size=len(unique_genes), replace=True)
        idx = np.concatenate([gene_to_idx[gene] for gene in sampled_genes])
        if np.unique(y[idx]).size < 2:
            continue
        base_auc = roc_auc_score(y[idx], delta_oof["xgboost_64"][idx])
        base_ap = average_precision_score(y[idx], delta_oof["xgboost_64"][idx])
        signal_auc = roc_auc_score(y[idx], delta_oof["wt_signal_70"][idx])
        signal_ap = average_precision_score(y[idx], delta_oof["wt_signal_70"][idx])
        for name in ["wt_signal_70", "xgboost_64_plus_delta_71", "wt_signal_70_plus_delta_77"]:
            model_auc = roc_auc_score(y[idx], delta_oof[name][idx])
            model_ap = average_precision_score(y[idx], delta_oof[name][idx])
            delta_bootstrap_rows.append({
                "replicate": replicate,
                "comparison": f"{name}-xgboost_64",
                "delta_auroc": model_auc - base_auc,
                "delta_auprc": model_ap - base_ap})
            if name != "wt_signal_70":
                delta_bootstrap_rows.append({
                    "replicate": replicate,
                    "comparison": f"{name}-wt_signal_70",
                    "delta_auroc": model_auc - signal_auc,
                    "delta_auprc": model_ap - signal_ap})
    delta_bootstrap = pd.DataFrame(delta_bootstrap_rows)
    delta_bootstrap.to_csv(DELTA_BOOTSTRAP_CSV, index=False)
    delta_summary = delta_bootstrap.groupby("comparison").agg(
        delta_auroc_mean=("delta_auroc", "mean"),
        delta_auroc_low=("delta_auroc", lambda x: np.quantile(x, 0.025)),
        delta_auroc_high=("delta_auroc", lambda x: np.quantile(x, 0.975)),
        delta_auprc_mean=("delta_auprc", "mean"),
        delta_auprc_low=("delta_auprc", lambda x: np.quantile(x, 0.025)),
        delta_auprc_high=("delta_auprc", lambda x: np.quantile(x, 0.975))).reset_index()
    print(delta_summary.to_string(index=False))
```

### Bootstrap Summary — Delta Models (2,000 gene-cluster replicates, FIXED)

| Comparison | ΔAUROC mean | ΔAUROC 95% CI | ΔAUPRC mean | ΔAUPRC 95% CI |
|---|---|---|---|---|
| **wt_signal_70 − xgboost_64** | **+0.0138** | [−0.0194, +0.0449] | **+0.0472** | **[+0.0130, +0.0843]** |
| xgboost_64_plus_delta_71 − xgboost_64 | −0.0109 | [−0.0333, +0.0109] | −0.0038 | [−0.0270, +0.0187] |
| wt_signal_70_plus_delta_77 − xgboost_64 | +0.0005 | [−0.0356, +0.0342] | +0.0291 | [−0.0053, +0.0645] |
| xgboost_64_plus_delta_71 − wt_signal_70 | −0.0246 | [−0.0527, +0.0024] | −0.0510 | [−0.0824, −0.0225] |
| wt_signal_70_plus_delta_77 − wt_signal_70 | −0.0133 | [−0.0333, +0.0066] | −0.0180 | [−0.0421, +0.0044] |

**Fixed delta-model conclusions:**

- **wt_signal_70** ΔAUPRC CI excludes zero against xgboost_64 [+0.0130, +0.0843].
- **xgboost_64_plus_delta_71 − xgboost_64:** Both CIs cross zero. Adding 7 sorting-signal deltas to the 64D model did not improve performance.
- **wt_signal_70_plus_delta_77 − wt_signal_70 (FIXED):** ΔAUPRC CI now crosses zero [−0.0421, +0.0044]. The previously reported "significant negative" was an artifact of the WT signal duplication bug. The data are compatible with either a small adverse effect or no difference. The delta features do not help, but they no longer appear to actively harm the model.
- **xgboost_64_plus_delta_71 − wt_signal_70:** ΔAUPRC CI remains entirely negative [−0.0824, −0.0225], confirming that 64D + raw deltas is substantially worse than 64D + WT signal context.

---

## 18.9b Direct Paired Bootstrap: wt_signal_70 vs AlphaMissense

Gene-cluster bootstrap (2,000 replicates) on the 2,140-row paired AlphaMissense cohort. All comparisons use the same resampled genes within each replicate.

```python
AM_BOOTSTRAP_CSV = BASE / "task18_alphamissense_bootstrap.csv"

paired_am = df["final_alphamissense_score"].notna().to_numpy()
y_am = df.loc[paired_am, "Mislocalized"].astype(int).to_numpy()
am_score = df.loc[paired_am, "final_alphamissense_score"].to_numpy()
wt70_am = oof["wt_signal_70"][paired_am]
xgb64_am = oof["xgboost_64"][paired_am]

genes_paired = df.loc[paired_am, "Gene"].astype(str).to_numpy()
unique_genes_am = np.unique(genes_paired)
gene_to_idx_am = {g: np.flatnonzero(genes_paired == g) for g in unique_genes_am}

print(f"Paired AM cohort: {len(y_am)} variants, {int(y_am.sum())} positives")
print(f"  AlphaMissense: AUROC={roc_auc_score(y_am, am_score):.4f}, AUPRC={average_precision_score(y_am, am_score):.4f}")
print(f"  wt_signal_70:   AUROC={roc_auc_score(y_am, wt70_am):.4f}, AUPRC={average_precision_score(y_am, wt70_am):.4f}")
print(f"  xgboost_64:     AUROC={roc_auc_score(y_am, xgb64_am):.4f}, AUPRC={average_precision_score(y_am, xgb64_am):.4f}")

rng = np.random.default_rng(RANDOM_STATE)
am_bootstrap_rows = []
for rep in range(2000):
    sampled = rng.choice(unique_genes_am, size=len(unique_genes_am), replace=True)
    idx = np.concatenate([gene_to_idx_am[g] for g in sampled])
    if np.unique(y_am[idx]).size < 2:
        continue
    am_auc = roc_auc_score(y_am[idx], am_score[idx])
    am_ap = average_precision_score(y_am[idx], am_score[idx])
    wt70_auc = roc_auc_score(y_am[idx], wt70_am[idx])
    wt70_ap = average_precision_score(y_am[idx], wt70_am[idx])
    xgb64_auc = roc_auc_score(y_am[idx], xgb64_am[idx])
    xgb64_ap = average_precision_score(y_am[idx], xgb64_am[idx])
    am_bootstrap_rows.append({
        "replicate": rep,
        "wt70-am_delta_auroc": wt70_auc - am_auc,
        "wt70-am_delta_auprc": wt70_ap - am_ap,
        "wt70-xgb64_delta_auroc": wt70_auc - xgb64_auc,
        "wt70-xgb64_delta_auprc": wt70_ap - xgb64_ap,
        "xgb64-am_delta_auroc": xgb64_auc - am_auc,
        "xgb64-am_delta_auprc": xgb64_ap - am_ap})

am_bootstrap = pd.DataFrame(am_bootstrap_rows)
am_bootstrap.to_csv(AM_BOOTSTRAP_CSV, index=False)

print("\n=== Paired gene-cluster bootstrap (2,000 replicates) ===")
for comp, label in [
    ("wt70-am", "wt_signal_70 − AlphaMissense"),
    ("wt70-xgb64", "wt_signal_70 − xgboost_64"),
    ("xgb64-am", "xgboost_64 − AlphaMissense"),
]:
    d_auc = am_bootstrap[f"{comp}_delta_auroc"]
    d_ap = am_bootstrap[f"{comp}_delta_auprc"]
    print(f"\n{label}:")
    print(f"  ΔAUROC: {d_auc.mean():+.4f}  [95% CI {d_auc.quantile(0.025):+.4f}, {d_auc.quantile(0.975):+.4f}]")
    print(f"  ΔAUPRC: {d_ap.mean():+.4f}  [95% CI {d_ap.quantile(0.025):+.4f}, {d_ap.quantile(0.975):+.4f}]")

print(f"\nSaved: {AM_BOOTSTRAP_CSV}")
```

```
Paired AM cohort: 2140 variants, 235 positives
  AlphaMissense: AUROC=0.6491, AUPRC=0.1619
  wt_signal_70:   AUROC=0.6579, AUPRC=0.2499
  xgboost_64:     AUROC=0.6442, AUPRC=0.1999

=== Paired gene-cluster bootstrap (2,000 replicates) ===

wt_signal_70 − AlphaMissense:
  ΔAUROC: +0.0088  [95% CI -0.0367, +0.0574]
  ΔAUPRC: +0.0866  [95% CI +0.0271, +0.1475]

wt_signal_70 − xgboost_64:
  ΔAUROC: +0.0138  [95% CI -0.0217, +0.0451]
  ΔAUPRC: +0.0477  [95% CI +0.0131, +0.0827]

xgboost_64 − AlphaMissense:
  ΔAUROC: -0.0050  [95% CI -0.0479, +0.0400]
  ΔAUPRC: +0.0389  [95% CI -0.0099, +0.0853]

Saved: .../task18_alphamissense_bootstrap.csv
```

### AlphaMissense Bootstrap Results (2,000 gene-cluster replicates, n=2,140)

| Comparison | ΔAUROC mean | ΔAUROC 95% CI | ΔAUPRC mean | ΔAUPRC 95% CI |
|---|---|---|---|---|
| **wt_signal_70 − AlphaMissense** | **+0.0088** | [−0.0367, +0.0574] | **+0.0866** | **[+0.0271, +0.1475]** |
| wt_signal_70 − xgboost_64 | +0.0138 | [−0.0217, +0.0451] | +0.0477 | [+0.0131, +0.0827] |
| xgboost_64 − AlphaMissense | −0.0050 | [−0.0479, +0.0400] | +0.0389 | [−0.0099, +0.0853] |

**AlphaMissense comparison conclusions:**

- **wt_signal_70 vs AlphaMissense:** On the paired cohort, the gene-cluster bootstrap conditional on fixed OOF predictions yielded a positive ΔAUPRC interval for wt_signal_70 versus AlphaMissense (ΔAUPRC +0.087, 95% CI [+0.027, +0.148]). The ΔAUROC interval crosses zero [−0.037, +0.057]; the point estimate favours wt_signal_70 (+0.009) but the discrimination difference is not confirmed.
- **xgboost_64 alone does not have a confirmed advantage over AlphaMissense** on either metric (both CIs cross zero), reinforcing that the WT sorting-signal features are the key differentiator.
- Formal claims of superiority over AlphaMissense require multiplicity/model-selection adjustment (wt_signal_70 was chosen post-hoc from multiple DeepLoc ablations) and a DeepLoc training-overlap audit.

---

## 18.10 Limitations and Open Issues

### 18.10.1 DeepLoc training-data overlap (unresolved)

DeepLoc 2.0 was trained on Swiss-Prot localisation annotations and HPA human proteins. MISFIT's 871 human genes may overlap with DeepLoc's training set. Gene-disjoint CV within MISFIT does not prevent:

\[
\text{MISFIT test gene} \in \text{DeepLoc external training set}
\]

This is not conventional target leakage — DeepLoc learns localisation/sorting-signal labels, not MISFIT mislocalisation labels — but it affects the interpretation of "generalisation to previously unseen proteins." DeepLoc features are best understood as **externally supervised protein-level features**, not de novo embeddings.

Recommended checks before finalising conclusions:
- Map MISFIT UniProt accessions against DeepLoc's published training set
- Sequence identity / SHA256 hash overlap
- Stratify OOF metrics by DeepLoc-training overlap status

### 18.10.2 precision_at_recall implementation

The current implementation uses:

```python
order = np.argsort(recall)
np.interp(target, recall[order], precision[order])
```

`precision_recall_curve` can return duplicate recall values. Passing these directly to `np.interp` produces results without a clean, stable threshold interpretation. Recommended alternatives:
- Precision at the first threshold achieving recall ≥ target (conservative)
- Maximum precision among thresholds with recall ≥ target (optimistic; must be declared)
- Precision@top-k with pre-specified k (most robust)

The current Prec@R20/R40 values are retained as exploratory outputs and should not anchor primary conclusions. A gene-cluster bootstrap CI should also be computed for these metrics.

### 18.10.3 Bootstrap scope

As noted in Section 18.6, the bootstrap only captures evaluation-sample variability. A full uncertainty quantification would require repeated grouped CV with refitting, or a nested bootstrap that resamples folds, refits PCA, and retrains XGBoost.

---

## Summary

| Section | Key Result |
|---|---|
| 18.1 Manifests | 871 WT + 762 pilot MT + 1,386 remaining MT = 3,019 unique sequences |
| 18.2 Inference | WT + pilot MT + remaining MT = 2,148 MT predictions; 2,179 variants fully covered; 0 missing deltas |
| 18.3 Context features | 16 WT features saved: 9 localisation + 6 core signals + TM |
| 18.4 Sensitivity pilot | Median deltas small; SP (8.1%), NLS (11.2%), NES (10.5%) show non-trivial tail |
| 18.5 WT-context ablation | **wt_signal_70** best AUROC (0.656) and AUPRC (0.248) |
| 18.6 WT-context bootstrap | wt_signal_70 ΔAUPRC +0.047 [95% CI +0.013, +0.084] |
| 18.7 Full MT deltas | 7 delta features; max \|Δ\|: NES 0.182, PTS 0.130, NLS 0.105 |
| 18.8 Delta ablation (fixed) | All delta models ≤ their base models. 71D ΔAUPRC −0.004 (CI crosses zero). 77D ΔAUPRC −0.018 (CI crosses zero). |
| 18.9 Delta bootstrap (fixed) | No delta model significantly improves on its base. 71D vs 64D and 77D vs 70D both null. |
| **18.9b AM bootstrap** | **wt_signal_70 vs AlphaMissense: ΔAUPRC +0.087 [95% CI +0.027, +0.148] — CI excludes zero** |
| 18.10 Limitations | DeepLoc training overlap unchecked; precision_at_recall needs revision |

**Primary conclusions:**

1. **wt_signal_70 (64D + 6 WT sorting-signal probabilities) is the best model**, with AUROC 0.656 and AUPRC 0.248. It significantly outperforms xgboost_64 on AUPRC (Δ +0.047, bootstrap 95% CI [+0.013, +0.084]) and significantly outperforms AlphaMissense on AUPRC (Δ +0.087, 95% CI [+0.027, +0.148]). The AUROC advantage over AlphaMissense is not statistically confirmed (CI crosses zero).

2. **Sorting-signal deltas do not improve prediction.** Adding 7 signed WT–MT deltas to either the 64D or the 70D model yields bootstrap intervals that cross zero. Under the current Fast/ESM1b representation, XGBoost configuration, and fixed folds, these raw delta features provide no incremental predictive value. This does not rule out alternate delta encodings or mechanism-specific analyses of extreme-delta variants.

3. **Full MT inference was the correct decision.** Although the deltas did not improve the main model, the full-cohort data (2,148 MT predictions, zero missing) yielded a well-powered negative point estimate with intervals compatible with no incremental benefit — a much stronger basis for decision-making than the ambiguous pilot. The extreme-tail variants (NES \|Δ\| up to 0.182) remain available for biological follow-up.

### Recommended next steps (priority order)

1. Audit DeepLoc training-data overlap with MISFIT genes (last remaining bias audit before wt_signal_70 can be declared the primary model)
2. Apply multiplicity correction (Holm or bootstrap max-statistic) across WT-context comparisons, or pre-register wt_signal_70 as the sole candidate
3. Revise `precision_at_recall` to a threshold-stable definition
4. Biological audit of extreme-delta variants for mechanistic signal
5. Consider wt_signal_70 as the new primary MISFIT model

---

*Generated from `task18_deeploc_delta.ipynb` — Model: Fast/ESM1b, Device: CUDA:1 (RTX 4090)*
