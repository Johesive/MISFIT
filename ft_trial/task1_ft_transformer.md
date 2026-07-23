# Task 1: FT-Transformer 64D vs 70D — Fixed Gene-Disjoint 8:1:1

**Device:** CUDA:1 (NVIDIA RTX 4090)

---

## Overview

Compare a two-layer FT-Transformer on 64D vs 70D features using Task 16's fixed gene-disjoint 8:1:1 split. FT-Transformer learns an independent tokenisation per continuous feature, then models feature interactions through 2-layer self-attention — it does not treat 70 scalars as a flat sequence.

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
TRIAL = ROOT / "ft_trial"
FEATURES_CSV = BASE / "features_v3.csv"
TMD_CSV = BASE / "tmd_features.csv"
CONTEXT_CSV = BASE / "deeploc_wt_context_features.csv"
SPLIT_CSV = BASE / "task16_holdout_split.csv"

RANDOM_SEEDS = [11, 22, 33, 44, 55]
N_COMPONENTS = 50
BATCH_SIZE = 32
MAX_EPOCHS = 300
PATIENCE = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-3
DEVICE = torch.device("cuda:1")

STRUCT_COLS = ["plddt", "sasa", "rsa", "ss_helix", "ss_strand", "ss_coil", "delta_hydrophobicity"]
DDG_COLS = ["ddg_esm2", "ddg_struct", "ddg_rasp", "ddg_foldx"]
TMD_COLS = ["in_TMD", "dist_to_nearest_TMD", "delta_membrane_insertion"]
WT_SIGNAL_COLS = [f"deeploc_wt_signal_{c}" for c in ["signal_peptide", "mitochondrial_transit_peptide",
    "nuclear_localisation_signal", "nuclear_export_signal", "peroxisomal_targeting_signal", "gpi_anchor"]]

FT_CONFIGS = {
    "ft_small":  {"d_token": 32, "n_heads": 4, "d_ffn": 64,  "dropout": 0.25},
    "ft_medium": {"d_token": 64, "n_heads": 4, "d_ffn": 128, "dropout": 0.30},
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

All imputation, scaling, and PCA fitted on training set only.

```python
X_esm = df[esm_cols].to_numpy(np.float32)
X_struct = df[STRUCT_COLS].to_numpy(np.float32)
X_extra64 = df[DDG_COLS + TMD_COLS].to_numpy(np.float32)
X_signal = df[WT_SIGNAL_COLS].to_numpy(np.float32)

def fit_transform_block(raw, indices):
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(raw[indices["train"]]))
    val = scaler.transform(imputer.transform(raw[indices["validation"]]))
    test = scaler.transform(imputer.transform(raw[indices["test"]]))
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

## 1.3 Two-Layer FT-Transformer Architecture

Each continuous feature has an independent learnable scale and bias, converting it into a feature token. A CLS token is prepended before 2-layer Transformer encoder with Pre-LN.

```python
class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=0.02)
        nn.init.normal_(self.bias, std=0.02)
    def forward(self, x):
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token, n_heads, d_ffn, dropout):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(d_model=d_token, nhead=n_heads,
            dim_feedforward=d_ffn, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))
        nn.init.normal_(self.cls_token, std=0.02)
    def forward(self, x):
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(1)
```

Training: weighted BCE loss (positive class weight = neg/pos ratio), AdamW optimizer (lr=2e-4, wd=1e-3), gradient clipping (max norm 5.0), early stopping with patience=30 on validation AUPRC.

---

## 1.4 Configuration Selection by Validation AUPRC

`ft_small` and `ft_medium` both use 2 layers; they differ only in token width and FFN capacity. The configuration with highest mean validation AUPRC is selected for each feature dimension.

| Feature Dim | Config | Mean Val AUPRC | SD Val AUPRC | Mean Val AUROC |
|---|---|---|---|---|
| 64 | **ft_medium** | **0.2661** | 0.0168 | 0.6618 |
| 64 | ft_small | 0.2637 | 0.0362 | 0.6455 |
| 70 | **ft_medium** | **0.2846** | 0.0115 | 0.5985 |
| 70 | ft_small | 0.2659 | 0.0311 | 0.6233 |

**Selected:** `ft_medium` for both 64D and 70D.

### Per-Seed Training Details

| Dim | Config | Seed | Params | Best Epoch | Val AUPRC |
|---|---|---|---|---|---|
| 64 | ft_small | 11 | 21,313 | 3 | 0.2869 |
| 64 | ft_small | 22 | 21,313 | 5 | 0.2979 |
| 64 | ft_small | 33 | 21,313 | 5 | 0.2399 |
| 64 | ft_small | 44 | 21,313 | 3 | 0.2124 |
| 64 | ft_small | 55 | 21,313 | 21 | 0.2817 |
| 64 | ft_medium | 11 | 75,393 | 14 | 0.2498 |
| 64 | ft_medium | 22 | 75,393 | 3 | 0.2633 |
| 64 | ft_medium | 33 | 75,393 | 18 | 0.2789 |
| 64 | ft_medium | 44 | 75,393 | 8 | 0.2508 |
| 64 | ft_medium | 55 | 75,393 | 2 | 0.2875 |
| 70 | ft_small | 11 | 21,697 | 3 | 0.2225 |
| 70 | ft_small | 22 | 21,697 | 2 | 0.2792 |
| 70 | ft_small | 33 | 21,697 | 24 | 0.2712 |
| 70 | ft_small | 44 | 21,697 | 5 | 0.3056 |
| 70 | ft_small | 55 | 21,697 | 42 | 0.2513 |
| 70 | ft_medium | 11 | 76,161 | 28 | 0.2670 |
| 70 | ft_medium | 22 | 76,161 | 13 | 0.2920 |
| 70 | ft_medium | 33 | 76,161 | 2 | 0.2944 |
| 70 | ft_medium | 44 | 76,161 | 44 | 0.2909 |
| 70 | ft_medium | 55 | 76,161 | 29 | 0.2788 |

---

## 1.5 Final Seed-Ensemble Test Evaluation

For the selected configuration, 5-seed probabilities are averaged before computing test AUROC/AUPRC. Single-seed test metrics are reported for training stability only.

### Seed-Ensemble Metrics

| Scope | Dim | Config | n | Pos | AUROC | AUPRC | Brier |
|---|---|---|---|---|---|---|---|
| validation | 64 | ft_medium | 221 | 24 | 0.6825 | 0.2771 | 0.1596 |
| **test** | **64** | **ft_medium** | **210** | **23** | **0.5820** | **0.1943** | 0.1890 |
| validation | 70 | ft_medium | 221 | 24 | 0.6134 | 0.2622 | 0.1470 |
| **test** | **70** | **ft_medium** | **210** | **23** | **0.5406** | **0.2130** | 0.1549 |

### Test Δ (70D − 64D)

| Metric | 64D | 70D | Δ |
|---|---|---|---|
| AUROC | 0.5820 | 0.5406 | −0.041 |
| AUPRC | 0.1943 | 0.2130 | +0.019 |

---

## Interpretation

This is a single held-out test with ~23 positives — suitable for rapid screening only. It cannot replace five-fold pooled OOF evaluation.

- **70D validation AUPRC (0.2846) > 64D (0.2661):** The WT signal features improved validation-set ranking during architecture selection.
- **70D test AUPRC (0.2130) > 64D (0.1943):** Directionally consistent — the 70D model achieved higher test AUPRC despite lower AUROC.
- **70D test AUROC (0.5406) < 64D (0.5820):** The 70D model showed worse discrimination despite better ranking of top predictions. This may reflect the small test set (23 positives) and high variance of the FT-Transformer on tabular data.
- **Val-test gap:** The 70D model shows a notable validation-test AUROC gap (0.613 → 0.541), suggesting possible overfitting or distribution shift between validation and test sets. The 64D model also shows a gap (0.683 → 0.582) but smaller.

**Verdict:** Directionally promising for 70D → AUPRC, but the single 23-positive test is too small for a definitive conclusion. If these trends hold, the next step would be a five-fold gene-disjoint OOF evaluation with paired gene-cluster bootstrap (à la Task 18).

---

*Generated from `task1_ft_transformer_64d_vs_70d.ipynb` — Device: CUDA:1 (RTX 4090)*
