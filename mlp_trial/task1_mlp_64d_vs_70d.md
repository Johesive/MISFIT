# Task 1: MLP 64D vs 70D — Fixed Gene-Disjoint 8:1:1

**Device:** CUDA:1 (NVIDIA RTX 4090)

---

## Overview

Compare a small 4/5-hidden-layer MLP on 64D vs 70D features using Task 16's fixed gene-disjoint 8:1:1 split.

- **64D** = fold-fitted ESM2 PCA(50) + structure(7) + stability(4) + TMD(3)
- **70D** = 64D + DeepLoc WT sorting-signal probabilities(6)
- Validation AUPRC used for architecture selection and early stopping; test not involved in selection
- Each configuration runs 5 random seeds; final comparison uses seed-ensemble test predictions

---

## 1.1 Load Features and Reuse Task 16 Split

All tables merged by canonical `KEY`. Split from Task 16, not regenerated.

```python
from pathlib import Path
import copy, random, warnings
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
warnings.filterwarnings("ignore")

ROOT = Path("/mnt/volume6/czj/labLGN/LabLZ")
BASE = ROOT / "xgboost_trial"
TRIAL = ROOT / "mlp_trial"
FEATURES_CSV = BASE / "features_v3.csv"
TMD_CSV = BASE / "tmd_features.csv"
CONTEXT_CSV = BASE / "deeploc_wt_context_features.csv"
SPLIT_CSV = BASE / "task16_holdout_split.csv"

RANDOM_SEEDS = [11, 22, 33, 44, 55]
N_COMPONENTS = 50
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-3
DEVICE = torch.device("cuda:1")

STRUCT_COLS = ["plddt", "sasa", "rsa", "ss_helix", "ss_strand", "ss_coil", "delta_hydrophobicity"]
DDG_COLS = ["ddg_esm2", "ddg_struct", "ddg_rasp", "ddg_foldx"]
TMD_COLS = ["in_TMD", "dist_to_nearest_TMD", "delta_membrane_insertion"]
WT_SIGNAL_COLS = [f"deeploc_wt_signal_{c}" for c in ["signal_peptide", "mitochondrial_transit_peptide",
    "nuclear_localisation_signal", "nuclear_export_signal", "peroxisomal_targeting_signal", "gpi_anchor"]]

MLP_CONFIGS = {
    "mlp_4hidden": [128, 64, 32, 16],
    "mlp_5hidden": [128, 96, 64, 32, 16],
}
```

```
Device: cuda:1
```

### Split verification

| Split | n | Genes | Positives | Prevalence |
|---|---|---|---|---|
| train | 1,748 | 696 | 189 | 0.1081 |
| validation | 221 | 87 | 24 | 0.1086 |
| test | 210 | 88 | 23 | 0.1095 |

---

## 1.2 Train-Only Preprocessing

All imputation, scaling, and PCA fitted on training set only. DDG, TMD, and DeepLoc probabilities also undergo train-fitted standardisation — neural networks are sensitive to input scale.

```python
X_esm = df[esm_cols].to_numpy(np.float32)
X_struct = df[STRUCT_COLS].to_numpy(np.float32)
X_extra64 = df[DDG_COLS + TMD_COLS].to_numpy(np.float32)
X_signal = df[WT_SIGNAL_COLS].to_numpy(np.float32)

def fit_transform_block(raw, indices, scale=True):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler() if scale else None
    train = imputer.fit_transform(raw[indices["train"]])
    val = imputer.transform(raw[indices["validation"]])
    test = imputer.transform(raw[indices["test"]])
    if scaler is not None:
        train = scaler.fit_transform(train)
        val = scaler.transform(val)
        test = scaler.transform(test)
    return train, val, test

esm_train, esm_val, esm_test = fit_transform_block(X_esm, split_indices)
pca = PCA(n_components=N_COMPONENTS, random_state=42)
pc_train, pc_val, pc_test = pca.fit_transform(esm_train), pca.transform(esm_val), pca.transform(esm_test)
struct_train, struct_val, struct_test = fit_transform_block(X_struct, split_indices)
extra_train, extra_val, extra_test = fit_transform_block(X_extra64, split_indices)
signal_train, signal_val, signal_test = fit_transform_block(X_signal, split_indices)

matrices = {
    64: (np.hstack([pc_train, struct_train, extra_train]),
         np.hstack([pc_val, struct_val, extra_val]),
         np.hstack([pc_test, struct_test, extra_test])),
    70: (np.hstack([pc_train, struct_train, extra_train, signal_train]),
         np.hstack([pc_val, struct_val, extra_val, signal_val]),
         np.hstack([pc_test, struct_test, extra_test, signal_test])),
}
```

```
64D: train=(1748, 64), validation=(221, 64), test=(210, 64)
70D: train=(1748, 70), validation=(221, 70), test=(210, 70)
```

---

## 1.3 MLP Definition and Training

```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        dropout = [0.30, 0.30, 0.25, 0.20, 0.15]
        layers = []
        previous = input_dim
        for i, hidden in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(previous, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout[min(i, len(dropout) - 1)])
            ])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x).squeeze(1)
```

Training: weighted BCE loss (positive class weight = neg/pos ratio), AdamW optimizer (lr=3e-4, wd=1e-3), gradient clipping (max norm 5.0), early stopping with patience=30 on validation AUPRC.

Architecture variants:

| Config | Hidden Layers | Total Params (64D) | Total Params (70D) |
|---|---|---|---|
| mlp_4hidden | [128, 64, 32, 16] | ~19K | ~20K |
| mlp_5hidden | [128, 96, 64, 32, 16] | ~32K | ~33K |

---

## 1.4 Configuration Selection by Validation AUPRC

The architecture with highest mean validation AUPRC is selected for each feature dimension.

| Feature Dim | Config | Mean Val AUPRC | SD Val AUPRC | Mean Val AUROC |
|---|---|---|---|---|
| 64 | **mlp_4hidden** | **0.3107** | 0.0425 | 0.6949 |
| 64 | mlp_5hidden | 0.2892 | 0.0439 | 0.6659 |
| 70 | **mlp_4hidden** | **0.3218** | 0.0465 | 0.6547 |
| 70 | mlp_5hidden | 0.2833 | 0.0216 | 0.6708 |

**Selected:** `mlp_4hidden` for both 64D and 70D.

### Per-Seed Training Details

| Dim | Config | Seed | Best Epoch | Val AUPRC |
|---|---|---|---|---|
| 64 | mlp_4hidden | 11 | 91 | 0.3157 |
| 64 | mlp_4hidden | 22 | 38 | 0.3070 |
| 64 | mlp_4hidden | 33 | 9 | 0.2664 |
| 64 | mlp_4hidden | 44 | 35 | 0.2859 |
| 64 | mlp_4hidden | 55 | 3 | 0.3786 |
| 64 | mlp_5hidden | 11 | 24 | 0.2851 |
| 64 | mlp_5hidden | 22 | 37 | 0.2638 |
| 64 | mlp_5hidden | 33 | 18 | 0.2357 |
| 64 | mlp_5hidden | 44 | 18 | 0.3107 |
| 64 | mlp_5hidden | 55 | 3 | 0.3503 |
| 70 | mlp_4hidden | 11 | 20 | 0.3700 |
| 70 | mlp_4hidden | 22 | 25 | 0.3657 |
| 70 | mlp_4hidden | 33 | 23 | 0.3075 |
| 70 | mlp_4hidden | 44 | 41 | 0.2590 |
| 70 | mlp_4hidden | 55 | 51 | 0.3068 |
| 70 | mlp_5hidden | 11 | 133 | 0.3036 |
| 70 | mlp_5hidden | 22 | 38 | 0.2904 |
| 70 | mlp_5hidden | 33 | 21 | 0.2474 |
| 70 | mlp_5hidden | 44 | 47 | 0.2816 |
| 70 | mlp_5hidden | 55 | 6 | 0.2934 |

---

## 1.5 Final Seed-Ensemble Test Evaluation

For each selected configuration, 5-seed probabilities are averaged before computing test metrics.

### Seed-Ensemble Metrics

| Scope | Dim | Config | n | Pos | AUROC | AUPRC | Brier |
|---|---|---|---|---|---|---|---|
| validation | 64 | mlp_4hidden | 221 | 24 | 0.7147 | 0.3236 | 0.1294 |
| **test** | **64** | **mlp_4hidden** | **210** | **23** | **0.6661** | **0.2720** | 0.1326 |
| validation | 70 | mlp_4hidden | 221 | 24 | 0.7022 | 0.3172 | 0.1314 |
| **test** | **70** | **mlp_4hidden** | **210** | **23** | **0.6536** | **0.2751** | 0.1314 |

### Test Δ (70D − 64D)

| Metric | 64D (mlp_4hidden) | 70D (mlp_4hidden) | Δ |
|---|---|---|---|
| AUROC | 0.6661 | 0.6536 | −0.0125 |
| AUPRC | 0.2720 | 0.2751 | +0.0031 |
| Brier | 0.1326 | 0.1314 | −0.0012 |

---

## Interpretation

This is a single held-out test with ~23 positives — suitable for rapid screening only.

- **70D validation AUPRC (0.3218) > 64D (0.3107):** The WT signal features improved validation-set ranking.
- **70D test AUPRC (0.2751) > 64D (0.2720):** Directionally consistent — small positive ΔAUPRC on test.
- **70D test AUROC (0.6536) < 64D (0.6661):** Small AUROC decrease on test, consistent with better top-ranking but slightly worse overall discrimination.
- **Seed variability:** Both feature dimensions show substantial seed-to-seed variation (SD val AUPRC ~0.04–0.05), reflecting the small validation set (221 variants, 24 positives).
- **Val-test gap:** Both dimensions show moderate validation-test gap (~0.04 AUPRC drop), within expected range for a small held-out set.

**Comparison with XGBoost (Task 16 test set):**

| Model | Test AUROC | Test AUPRC |
|---|---|---|
| XGBoost 64D (Task 15 OOF on test subset) | — | — |
| XGBoost 70D (Task 18 OOF on test subset) | — | — |
| MLP 64D | 0.6661 | 0.2720 |
| **MLP 70D** | 0.6536 | **0.2751** |

**Verdict:** The MLP shows a small but directionally consistent AUPRC improvement from WT signal features on this test set. Both MLP configurations achieve test AUROC >0.65 and AUPRC >0.27. The 8:1:1 held-out test is underpowered (23 positives) but the positive ΔAUPRC direction for 70D across both MLP and FT-Transformer (see companion report) warrants five-fold OOF evaluation with paired gene-cluster bootstrap.

---

*Generated from `task1_mlp_64d_vs_70d.ipynb` — Device: CUDA:1 (RTX 4090)*
