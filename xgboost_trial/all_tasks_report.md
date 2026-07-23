# 错定位预测模型完整分析报告

## 当前权威状态（更新于 2026-07-23）

本节记录经过本轮数据审查和代码修正后的数据口径、pipeline 与结果。后面的 Task 1–Task 6 为早期实验记录，其中部分标签、特征维度和实现已经被修正，不能再作为当前模型结论。

### 0. 冻结的 primary model：XGBoost 70D

自 2026-07-23 起，MISFIT 的 primary model 冻结为 **XGBoost 70D (`wt_signal_70`)**。此前的 64D XGBoost 保留为 ablation baseline；Task 17 中“保留 64D”的结论是当时尚无 DeepLoc features 时的历史决策，已被 Task 18 和后续固定 8:1:1 benchmark 取代。

#### Primary-model feature contract

| Feature group | Dimensions | Frozen definition |
|---|---:|---|
| ESM2 local WT–MT delta | 50 | 1280D local delta embedding；仅在 training fold/split 内完成 median imputation、standardisation 和 PCA(50) |
| Structure | 7 | `plddt`, `sasa`, `rsa`, `ss_helix`, `ss_strand`, `ss_coil`, `delta_hydrophobicity` |
| Stability-related | 4 | `ddg_esm2`, `ddg_struct`, `ddg_rasp`, `ddg_foldx` |
| TMD | 3 | `in_TMD`, `dist_to_nearest_TMD`, `delta_membrane_insertion` |
| DeepLoc WT sorting context | 6 | Fast/ESM1b WT probabilities：signal peptide、mitochondrial transit peptide、NLS、NES、PTS、GPI anchor |
| **Total** | **70** | 不包含 WT localisation probabilities、DeepLoc TM probability、plant-specific signals 或 WT–MT DeepLoc deltas |

同一 protein/gene 的 variants 共享六个 WT sorting-signal probabilities；它们是 externally supervised protein-level context。64D部分为 variant-level representation。所有 joins 必须使用 canonical `KEY = Gene + "||" + Mutation_used`；禁止依赖 row order。

#### Frozen XGBoost hyperparameters

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

Training 使用 `compute_sample_weight("balanced", y_train)`。后续不得根据现有 OOF 或 fixed test 结果调整 feature set、split、hyperparameters 或 random state 后仍将其描述为同一个冻结模型；任何改变必须使用新的 model/version 名称。

以上冻结的是实际 notebook 显式传入的 estimator arguments。原始服务器运行没有记录 `xgboost`、`scikit-learn`、Python 与 CUDA 的完整版本，因此目前无法承诺跨环境 byte-for-byte identical predictions。发布或生成最终 artefact 前应导出服务器 environment；若依赖版本变化，即使参数名相同，也应做 prediction checksum/reproduction check。

#### Authoritative primary-model results

| Evaluation | Model | n / positives | AUROC | AUPRC |
|---|---|---:|---:|---:|
| Five-fold gene-disjoint pooled OOF | XGB64 | 2179 / 236 | 0.6422 | 0.1981 |
| Five-fold gene-disjoint pooled OOF | **XGB70** | 2179 / 236 | **0.6560** | **0.2479** |
| Fixed gene-disjoint 8:1:1 test | XGB64 | 210 / 23 | 0.6266 | 0.2075 |
| Fixed gene-disjoint 8:1:1 test | **XGB70** | 210 / 23 | **0.7012** | **0.2843** |

Paired gene-cluster bootstrap（2,000 replicates，conditional on fixed predictions）：

| Comparison | ΔAUROC mean (95% CI) | ΔAUPRC mean (95% CI) |
|---|---:|---:|
| XGB70 − XGB64，pooled OOF | +0.0138 (−0.0194, +0.0449) | +0.0472 (+0.0130, +0.0843) |
| XGB70 − XGB64，fixed test | +0.073 (+0.008, +0.140) | +0.072 (−0.089, +0.205) |

Primary-model selection 是基于 pooled OOF、fixed-test point estimates、稳定性、计算效率和可解释性的综合 pragmatic decision。Bootstrap 不包含 split、PCA、retraining 或 model-selection uncertainty；fixed test 已在多轮模型比较中使用，不能描述为 untouched independent validation。

神经网络筛选没有改变该决定：MLP64/70 在 fixed test 上分别为 0.6661/0.2720 和 0.6536/0.2751（AUROC/AUPRC），没有通过 paired bootstrap 显示相对 XGBoost 的可靠增量；FT-Transformer 更弱。因此 MLP 与 FT-Transformer 不继续推进。

### 1. 当前研究任务与数据

- 当前任务为二分类：预测 single amino acid variant 是否导致 protein mislocalisation。
- 总样本数：2179，其中主表 2089，Additional Benign Variants 90。
- 总基因数：871。
- 负例：1943；正例：236；prevalence/base rate = 0.1083。
- 当前二分类 target 直接使用 `Mislocalized`，不再用 `df["label_5class"].notna()` 或 `reloc_v3 > 0` 间接生成。后两种写法会把标签是否缺失与二分类生物学定义混在一起。
- 多分类标签仅用于描述错定位 phenotype；C0 为不重定位，C1–C5 为不同错定位类型。

| Class | Phenotype | n |
|---|---|---:|
| C0 | 不重定位 | 1943 |
| C1 | 同区室 | 13 |
| C2 | 聚集 | 34 |
| C3 | 分泌途径 | 121 |
| C4 | 核定位 | 29 |
| C5 | 细胞质 | 39 |
| C1–C5 | 二分类正例合计 | 236 |

Additional Benign Variants 与 Variant Annotation 的原始表结构不同，因此在 data preparation 阶段显式构建标签。已人工核对并将 `RPE65 K294T` 归入 `C5_cytoplasmic`；其 `Mislocalized = 1`。这一修改写入：

- `data_preparation/3.5.1_classifier_label.ipynb`
- `data_preparation/完整代码.py`

### 2. Variant identifier 修正

当前 canonical identifier 为：

```python
KEY = Gene + "||" + Mutation_used
```

`Mutation_used` 是经过 data preparation 统一和校验、供模型 pipeline 使用的突变表示。2179 个样本的 `KEY` 均唯一。

旧 ESM2 pickle 的 key 是历史格式 `Gene||Variant`。在已有 cache 重新生成前，Task 0 仅为读取旧 pickle 保留：

```python
legacy_esm_key = Gene + "||" + Variant
```

该 legacy key 只允许用于 ESM2 cache lookup，不能成为导出特征表的 identifier。canonical `KEY` 与 legacy key 仅有 88 行相同；此前直接用 `Gene||Mutation_used` 查询旧 cache，导致特征矩阵错误缩减到 88 行。修正后的 Task 0 会要求最终保留全部 2179 行，并在 ESM embedding 缺失时直接报错。

### 3. 当前特征定义与完整性

基础模型使用：

- ESM2 embedding：1280 维；随后在每个 CV training fold 内拟合 PCA，保留 50 PCs。
- 结构特征：7 维，分别为 `plddt`, `sasa`, `rsa`, `ss_helix`, `ss_strand`, `ss_coil`, `delta_hydrophobicity`。
- stability 特征：`ddg_esm2`, `ddg_struct`, `ddg_rasp`, `ddg_foldx`。
- TMD 特征：3 维，分别为 `in_TMD`, `dist_to_nearest_TMD`, `delta_membrane_insertion`。

当前本地数据可用性：

| Feature | 非缺失/总数 |
|---|---:|
| 7 个结构特征整体 | 2179/2179 |
| `ddg_esm2` | 2179/2179 |
| `ddg_struct` | 2168/2179 |
| `ddg_rasp` | 2168/2179 |
| `ddg_foldx` | 2166/2179 |
| 3 个 TMD 特征的 key match | 2179/2179 |

TMD 信号分布：`in_TMD` 非零 151 行，`dist_to_nearest_TMD` 非零 664 行，`delta_membrane_insertion` 非零 151 行。

所有 imputer、scaler 和 PCA 都必须只在每个 training fold 上拟合，随后转换对应 validation fold，防止 data leakage。CV 使用 gene-grouped split，防止同一 gene 同时进入 training 和 validation。

### 4. AlphaMissense 缺失值补全

新增 `task0.5_complete_alphamissense.ipynb`，放在 Task 0 特征构建之后、AlphaMissense baseline 之前。其逻辑为：

1. 保留原有 AlphaMissense score；
2. 对缺失且属于 nonsynonymous mutation 的行，依据 protein accession 查询 AlphaFold DB metadata API，并读取 `amAnnotationsUrl`；
3. synonymous mutation 标为 `not_applicable_synonymous`，不伪造分数；
4. 记录每行 retrieval status，失败或无 annotation URL 的样本保持缺失；
5. 导出 `alphamissense_completed.csv`，identifier 使用 canonical `KEY`。

运行结果：

- 原始有分数：2053。
- 原始缺失：126，其中 nonsynonymous 95，synonymous 31。
- 成功补回：87。
- `no_annotation_url`：7。
- query error：1（protein accession `P07203`）。
- 最终可用 AlphaMissense score：2140/2179。

最终 status 分布为：`original` 2053、`retrieved` 87、`not_applicable_synonymous` 31、`no_annotation_url` 7、query error 1。没有对剩余缺失值作数值插补；AlphaMissense 对比只在 score 实际可用的 paired cohort 上进行。

### 5. Task 12：61 维 stability ablation（当前已运行结果）

`task12_ddg_ablation.ipynb` 已重写。所有配置共享同一组 grouped CV folds，并在 fold 内执行预处理：

| 配置 | 维度 | 组成 |
|---|---:|---|
| `baseline_pca_struct` | 57 | PCA(50) + structure(7) |
| `plus_ddg_esm2` | 58 | 57D + `ddg_esm2` |
| `plus_structure_ddgs` | 60 | 58D + `ddg_struct` + `ddg_rasp` |
| `final_all_ddgs` | 61 | 60D + `ddg_foldx` |

Full-cohort out-of-fold results：

| Model | n | Positive | AUROC | AUPRC | ΔAUROC vs 57D |
|---|---:|---:|---:|---:|---:|
| 57D baseline | 2179 | 236 | 0.5898 | 0.1567 | 0.0000 |
| 58D + `ddg_esm2` | 2179 | 236 | 0.6112 | 0.1688 | +0.0214 |
| 60D + structure-derived ddGs | 2179 | 236 | 0.6038 | 0.1697 | +0.0140 |
| 61D all ddGs | 2179 | 236 | 0.6286 | 0.1758 | +0.0388 |

Paired AlphaMissense comparison（同一 2140 个样本，positive = 235，prevalence = 0.1098）：

| Predictor | AUROC | AUPRC |
|---|---:|---:|
| AlphaMissense | 0.6491 | 0.1619 |
| MISFIT 61D | 0.6311 | 0.1780 |
| MISFIT − AlphaMissense | −0.0179 | +0.0160 |

61D 模型中 stability-related feature importance：

| Feature | Importance | Rank |
|---|---:|---:|
| `ddg_esm2` | 0.036686 | 1 |
| `ddg_rasp` | 0.022989 | 2 |
| `ddg_foldx` | 0.016574 | 26 |
| `ddg_struct` | 0.016096 | 31 |

当前可支持的结论：stability features 在这组 OOF 结果中提供了有希望的增量信号，61D 配置的 AUROC 和 AUPRC 均高于 57D baseline。与 AlphaMissense 比较时，AlphaMissense AUROC 更高，MISFIT 61D AUPRC 更高。尚未完成 paired confidence interval 或 statistical test，因此不能据此宣称任一模型显著优于另一模型；feature importance 也不能直接解释为因果贡献。

### 6. Task 15：61D vs 64D TMD 增量实验（已运行）

TMD 增量实验应在 `task15_full64.ipynb` 中完成；Task 14 只负责生成 TMD features，已恢复为原用途。

Task 15 使用 Task 12 的相同 folds 和固定 61D OOF prediction，评估加入 3 个 TMD features 后的增量：

- 固定复用 Task 12 的 folds 和 61D OOF prediction；
- 64D = 61D + `in_TMD` + `dist_to_nearest_TMD` + `delta_membrane_insertion`；
- 对 TMD 特征同样执行 fold-local preprocessing；
- 在 full cohort 和 AlphaMissense paired cohort 上比较 61D 与 64D；
- 保存 OOF predictions、metrics 和 feature importance。

各 fold AUROC：

| Fold | 61D | 64D | ΔAUROC |
|---:|---:|---:|---:|
| 0 | 0.6459 | 0.6608 | +0.0149 |
| 1 | 0.5959 | 0.6314 | +0.0355 |
| 2 | 0.5924 | 0.6483 | +0.0559 |
| 3 | 0.6416 | 0.6344 | −0.0072 |
| 4 | 0.6976 | 0.6743 | −0.0233 |

Full-cohort OOF comparison（n = 2179，positive = 236，prevalence = 0.1083）：

| Model | AUROC | AUPRC |
|---|---:|---:|
| Stability 61D | 0.6286 | 0.1758 |
| Stability + TMD 64D | 0.6422 | 0.1981 |
| 64D − 61D | +0.0135 | +0.0223 |

Paired AlphaMissense cohort（n = 2140，positive = 235，prevalence = 0.1098）：

| Predictor | AUROC | AUPRC |
|---|---:|---:|
| AlphaMissense | 0.6491 | 0.1619 |
| Stability 61D | 0.6311 | 0.1780 |
| Stability + TMD 64D | 0.6442 | 0.1999 |
| 64D − AlphaMissense | −0.0048 | +0.0380 |

64D 模型中的 DDG/TMD importance ranks：

| Feature | Importance | Rank |
|---|---:|---:|
| `in_TMD` | 0.036551 | 1 |
| `ddg_esm2` | 0.036359 | 2 |
| `dist_to_nearest_TMD` | 0.029027 | 3 |
| `delta_membrane_insertion` | 0.026835 | 4 |
| `ddg_rasp` | 0.019963 | 7 |
| `ddg_struct` | 0.015078 | 29 |
| `ddg_foldx` | 0.013715 | 36 |

当前结果支持 TMD features 具有进一步研究价值：64D 的 pooled OOF AUROC 和 AUPRC 均高于固定 61D，且三个 TMD features 的 importance 均位于前四名中的三个位置。不过，fold 3 和 fold 4 的 AUROC 下降，说明增益并非跨 fold 一致。尚无 paired bootstrap confidence interval 或 permutation test，因此不能将 `+0.0135` 和 `+0.0223` 描述为统计显著提升。与 AlphaMissense 相比，64D 的 AUROC 已接近，但仍低 0.0048；AUPRC 高 0.0380。

### 7. Task 16：XGBoost vs TabPFN（8:1:1 held-out test，已运行）

Task 16 使用 gene-disjoint 8:1:1 split，在相同 64D features 上比较 XGBoost、TabPFN 和 ensemble。Validation 选择的 TabPFN ensemble weight 为 0.60；最终评价只使用 held-out test。

Full held-out test（n = 210，positive = 23，prevalence = 0.1095）：

| Model | AUROC | AUPRC |
|---|---:|---:|
| XGBoost 64D | 0.6266 | 0.2075 |
| TabPFN 64D | 0.6002 | 0.2441 |
| Ensemble 64D | 0.6223 | 0.2056 |

Paired AlphaMissense test subset（n = 204，positive = 22，prevalence = 0.1078）：

| Predictor | AUROC | AUPRC |
|---|---:|---:|
| AlphaMissense | 0.6190 | 0.1552 |
| XGBoost 64D | 0.6391 | 0.2113 |
| TabPFN 64D | 0.6155 | 0.2500 |
| Ensemble 64D | 0.6380 | 0.2096 |

当前解释：

- XGBoost 的 discrimination ranking 更稳定，在 full test 上 AUROC 比 TabPFN 高 0.0264。
- TabPFN 的 full-test AUPRC 比 XGBoost 高 0.0366，说明它在正例优先排序方面可能具有互补信号；但 test 中只有 23 个正例，该差值的不确定性可能很大。
- Validation 选择的 0.60 TabPFN ensemble 没有在 test 上超过 XGBoost：AUROC 低 0.0043，AUPRC 低 0.0019。因此当前不采用这个 ensemble。
- 在 paired AlphaMissense test subset 上，XGBoost AUROC 和 AUPRC 均高于 AlphaMissense；TabPFN 的 AUPRC 最高，但 AUROC 略低于 AlphaMissense。该结果只适用于 204 行单次 held-out subset，不能替代 Task 15 的 2140 行 pooled OOF comparison。
- Task 15 与 Task 16 的指标不可直接当作模型性能变化：前者是 5-fold pooled OOF，后者是单次约 10% held-out test。若要正式判断 XGBoost 与 TabPFN 的差异，需要对 test predictions 做 gene-cluster bootstrap confidence intervals，或运行 repeated gene-grouped holdout。

### 8. Task 17：稳健性、模型决策与 error analysis（已运行）

#### Gene-cluster bootstrap

Task 15 full-cohort OOF 的 paired bootstrap differences：

| Comparison | ΔAUROC mean (95% CI) | ΔAUPRC mean (95% CI) |
|---|---:|---:|
| 64D − 61D | +0.0133 (−0.0081, 0.0346) | +0.0229 (−0.0050, 0.0508) |
| 64D − AlphaMissense, paired cohort | −0.0049 (−0.0481, 0.0379) | +0.0388 (−0.0063, 0.0862) |

Task 16 paired held-out test differences：

| Comparison | ΔAUROC mean (95% CI) | ΔAUPRC mean (95% CI) |
|---|---:|---:|
| TabPFN − XGBoost | −0.0216 (−0.1111, 0.0630) | +0.0394 (−0.0815, 0.1647) |
| Ensemble − XGBoost | −0.0007 (−0.0555, 0.0522) | +0.0008 (−0.0649, 0.0643) |
| XGBoost − AlphaMissense | +0.0182 (−0.1455, 0.1823) | +0.0523 (−0.0816, 0.2057) |

所有 paired difference CI 都跨越 0。现有数据不支持 64D 显著优于 61D、不支持 MISFIT 显著优于 AlphaMissense，也不支持 TabPFN 或 ensemble 显著优于 XGBoost。**这是 Task 17 当时的历史决策**：在 DeepLoc experiment 完成前暂时保留 XGBoost 64D。当前 primary model 已由上方冻结规格更新为 XGBoost 70D；本段不能再作为当前模型选择结论引用。

#### TMD robustness

| Subgroup | n | Positive | 61D AUROC | 64D AUROC | ΔAUROC | 61D AUPRC | 64D AUPRC | ΔAUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 2179 | 236 | 0.6286 | 0.6422 | +0.0135 | 0.1758 | 0.1981 | +0.0223 |
| `in_TMD` | 151 | 39 | 0.7035 | 0.6758 | −0.0277 | 0.4440 | 0.3926 | −0.0514 |
| `outside_TMD` | 2028 | 197 | 0.6074 | 0.6158 | +0.0084 | 0.1480 | 0.1623 | +0.0143 |

64D 的 fold-level AUROC 在 fold 0–2 提升，在 fold 3–4 下降；AUPRC 仅在 fold 3 下降。最重要的反常结果是：TMD features 在真正的 `in_TMD` subgroup 上同时降低 AUROC 和 AUPRC，整体增益来自数量占绝对多数的 `outside_TMD` 样本。多个 prediction change 最大的 genes 只有 1–4 个 variants，表明 feature importance 排名可能受少量 genes 或稀疏特征影响。因此，当前不能将总体提升解释为“模型更好地识别了 TMD variants”；需要进一步检查 TMD 编码、distance feature 的含义以及 gene-specific influence。

#### Exploratory error analysis

使用 full OOF predictions 上 F1 最大化得到探索性 threshold = 0.277045：TN = 1506、FP = 437、FN = 134、TP = 102，对应 sensitivity = 0.4322、specificity = 0.7751、precision = 0.1892。该 threshold 在同一 OOF 数据上选择，只用于描述错误结构，不能作为独立性能结果。

- `in_TMD`：sensitivity = 0.7692、specificity = 0.5446、precision = 0.3704；模型更容易召回 TMD 正例，但产生较多 TMD false positives。
- `outside_TMD`：sensitivity = 0.3655、specificity = 0.7892、precision = 0.1572；主要问题是漏掉正例。
- fold 3 sensitivity 最低（0.3333）；fold 4 precision 最低（0.1374），再次表明 split sensitivity 明显。
- Additional Benign Variants 有 9 FP；其中唯一正例被识别。由于该来源只有一个正例，不能对 sensitivity 作稳定推断。
- AlphaMissense 缺失组只有一个正例且被漏判，样本量不足以判断 missingness effect。

### 9. 本轮 notebook/file 变更清单

| 文件 | 当前作用 |
|---|---|
| `task0_build_features_v3.ipynb` | 构建 2179 行 features；导出 canonical `KEY`；legacy key 仅查询旧 ESM cache |
| `task0.5_complete_alphamissense.ipynb` | 补取缺失 AlphaMissense annotations 并记录 status |
| `task1_alphamissense_baseline.ipynb` | 在 score 可用的 paired cohort 上计算 AlphaMissense baseline |
| `task12_ddg_ablation.ipynb` | 57/58/60/61D stability ablation 与 paired AlphaMissense 比较 |
| `task14_tmd_features.ipynb` | 生成 3 个 TMD features；保持原职责 |
| `task15_full64.ipynb` | 使用相同 folds 对比固定 61D 与加入 TMD 后的 64D |
| `task16_tabpfn.ipynb` | 使用 gene-disjoint 8:1:1 holdout，在统一 64D 数据口径下比较 XGBoost、TabPFN 与 validation-selected ensemble |
| `task17_robustness_error_analysis.ipynb` | Gene-cluster bootstrap、主模型决策、TMD 稳健性和 exploratory error analysis |
| `task18_deeploc_delta.ipynb` | DeepLoc WT context、全量 WT–MT inference、64D/70D/delta ablation 与 paired bootstrap |
| `task18.md` | Task 18 代码、输出、限制及修正后的解释 |
| `task19_deeploc_overlap_audit.ipynb` | Current UniProt annotation-status exploratory audit；不能替代 actual DeepLoc training-set exact-overlap audit |
| `../mlp_trial/task1_mlp_64d_vs_70d.ipynb` | 固定 8:1:1 split 上的 MLP 64D/70D screening |
| `../ft_trial/task1_ft_transformer_64d_vs_70d.ipynb` | 固定 8:1:1 split 上的 FT-Transformer 64D/70D screening |

Task 0 → Task 19、MLP/FT screening 与固定 8:1:1 XGB70 benchmark 已完成。模型侧冻结 XGBoost 70D，不继续推进当前 MLP、FT-Transformer、TabPFN ensemble 或 DeepLoc delta models。下一步应优先进行 biological subgroup/error analysis、异常标签审查和 external/prospective validation；现有 fixed test 不再用于新一轮 hyperparameter tuning。

---

## 历史实验记录（已被上述修正口径取代）

以下内容保留用于追踪研究过程。凡涉及 `reloc_v3 > 0`、222 个正例、1288 维旧特征定义、非 canonical identifier 或旧 CV/preprocessing 的结果，均不应引用为当前最终结果。

## Task 1: AlphaMissense 同折基线 —— "我们要打败的那个数"

**目标**: 在 AlphaMissense 分数不为 NaN 的子集 (AM 子集) 上，直接比较 AlphaMissense 分数 vs v3 XGBoost 模型。

### 代码

```python
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

BASE_PATH = "/mnt/volume6/czj/labLGN/LabLZ/"

# ===== 加载数据 =====
df_main = pd.read_csv(BASE_PATH + "cell2024_model_single_subst.csv")
df_feat = pd.read_csv(BASE_PATH + "features_v3.csv")

# 对齐校验
assert len(df_main) == len(df_feat) == 2179
assert (df_main["Gene"].values == df_feat["Gene"].values).all()

# 把 source / AlphaMissense 从主表合并到特征表
df_feat["source"] = df_main["source"].values
df_feat["AlphaMissense_score"] = df_main["AlphaMissense score"].values

# 特征列定义
ID_COLS = ["KEY", "Gene", "reloc_v3"]
META_COLS = ["source", "AlphaMissense_score"]
NAN_PLACEHOLDERS = ["ddg", "plddt_diff"]
exclude_cols = ID_COLS + META_COLS + NAN_PLACEHOLDERS
feat_cols = [c for c in df_feat.columns if c not in exclude_cols]

X_full = df_feat[feat_cols].values.astype(np.float32)
y_5class = df_feat["reloc_v3"].values.astype(int)
y_bin = (y_5class > 0).astype(int)
groups = df_feat["Gene"].values
source_arr = df_feat["source"].values
am_score_arr = df_feat["AlphaMissense_score"].values.astype(np.float64)

n_total = len(y_bin)
n_pos = int(y_bin.sum())
n_neg = n_total - n_pos
base_rate = n_pos / n_total

print(f"数据加载完毕: n={n_total}, 正例={n_pos}, 负例={n_neg}, base_rate={base_rate:.4f}")
print(f"source 分布: {dict(zip(*np.unique(source_arr, return_counts=True)))}")
print(f"特征列数: {len(feat_cols)}")

# 统一 5 折 CV
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

# ===== CV 工具函数 =====
def cv_evaluate_binary(X, y, groups, sample_weight_mode="balanced"):
    xgb_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.5,
        objective="binary:logistic", eval_metric="aucpr",
        random_state=42, n_jobs=-1, tree_method="hist",
    )
    oof = np.zeros(len(y), dtype=np.float32)
    per_fold = []
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_tr = scl.fit_transform(imp.fit_transform(X_tr_raw)).astype(np.float32)
        X_te = scl.transform(imp.transform(X_te_raw)).astype(np.float32)
        sw = compute_sample_weight("balanced", y_tr)
        model = XGBClassifier(**xgb_params)
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        oof[te_idx] = model.predict_proba(X_te)[:, 1]
        y_te = y[te_idx]
        per_fold.append({
            "fold": fold,
            "auroc": roc_auc_score(y_te, oof[te_idx]),
            "auprc": average_precision_score(y_te, oof[te_idx]),
            "n": len(y_te), "n_pos": int(y_te.sum())
        })
    return oof, per_fold

def print_metrics(label, y_true, oof, per_fold=None):
    auroc = roc_auc_score(y_true, oof)
    auprc = average_precision_score(y_true, oof)
    n = len(y_true); n_pos = int(y_true.sum()); br = n_pos / n
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  n={n}, pos={n_pos}, base_rate={br:.4f}")
    print(f"  AUROC = {auroc:.4f}")
    print(f"  AUPRC = {auprc:.4f}  (base_rate={br:.4f})")
    if per_fold:
        fa = [f["auroc"] for f in per_fold]
        fp = [f["auprc"] for f in per_fold]
        print(f"  Per-fold AUROC: {[f'{v:.3f}' for v in fa]}  "
              f"mean={np.mean(fa):.4f} ± {np.std(fa):.4f}")
        print(f"  Per-fold AUPRC: {[f'{v:.3f}' for v in fp]}  "
              f"mean={np.mean(fp):.4f} ± {np.std(fp):.4f}")
    return {"label": label, "n": n, "n_pos": n_pos, "base_rate": br,
            "auroc": auroc, "auprc": auprc}
```

### 输出

```
数据加载完毕: n=2179, 正例=222, 负例=1957, base_rate=0.1019
source 分布: {'additional_benign': np.int64(90), 'main': np.int64(2089)}
特征列数: 1288
统一 CV 已初始化 (StratifiedGroupKFold, n_splits=5, groups=Gene)
CV 工具函数就绪
```

### 1a. AlphaMissense 分数直接作为预测值

```python
# ===== AM 子集 =====
am_mask = np.isfinite(am_score_arr)
X_am = X_full[am_mask]
y_am = y_bin[am_mask]
g_am = groups[am_mask]
am_v = am_score_arr[am_mask]

n_am = len(y_am)
pos_am = int(y_am.sum())
print(f"AM 子集: n={n_am}, 正例={pos_am}, base_rate={pos_am/n_am:.4f}")
print(f"(全量 n={n_total}, AM 子集覆盖 {n_am}/{n_total} = {n_am/n_total*100:.1f}%)")

# AlphaMissense 直接打分
am_auroc = roc_auc_score(y_am, am_v)
am_auprc = average_precision_score(y_am, am_v)
print(f"\nAlphaMissense-alone (直接作为预测值):")
print(f"  n={n_am}, pos={pos_am}, base_rate={pos_am/n_am:.4f}")
print(f"  AUROC = {am_auroc:.4f}")
print(f"  AUPRC = {am_auprc:.4f}")
```

```
AM 子集: n=2053, 正例=221, base_rate=0.1076
(全量 n=2179, AM 子集覆盖 2053/2179 = 94.2%)

AlphaMissense-alone (直接作为预测值):
  n=2053, pos=221, base_rate=0.1076
  AUROC = 0.6374
  AUPRC = 0.1547
```

### 1b. v3 XGBoost 在 AM 子集上训练+评估

```python
print("在 AM 子集上用统一 CV 训练 XGBoost ...")
oof_am_cv, folds_am_cv = cv_evaluate_binary(X_am, y_am, g_am)
r_am_cv = print_metrics("v3 XGBoost (AM 子集 CV)", y_am, oof_am_cv, folds_am_cv)
```

```
============================================================
  v3 XGBoost (AM 子集 CV)
  n=2053, pos=221, base_rate=0.1076
  AUROC = 0.5719
  AUPRC = 0.1527  (base_rate=0.1076)
  Per-fold AUROC: ['0.604', '0.553', '0.618', '0.548', '0.557']  mean=0.5761 ± 0.0292
  Per-fold AUPRC: ['0.152', '0.126', '0.245', '0.167', '0.143']  mean=0.1666 ± 0.0415
```

### 1c. v3 XGBoost 全量训练，仅在 AM 子集行上评估

```python
print("在全量数据上用统一 CV 训练 XGBoost ...")
oof_full, folds_full = cv_evaluate_binary(X_full, y_bin, groups)

# 只在 AM 子集行上评估
oof_full_on_am = oof_full[am_mask]
r_full_on_am = print_metrics("v3 XGBoost (全量训练, 仅在 AM 子集评估)",
                              y_am, oof_full_on_am)
```

```
============================================================
  v3 XGBoost (全量训练, 仅在 AM 子集评估)
  n=2053, pos=221, base_rate=0.1076
  AUROC = 0.5590
  AUPRC = 0.1374  (base_rate=0.1076)
```

### Task 1 汇总

| 方法 | AUROC | AUPRC |
|---|---|---|
| **AlphaMissense-alone (直接打分)** | **0.6374** | 0.1547 |
| v3 XGBoost (AM 子集 CV 训练) | 0.5719 | 0.1527 |
| v3 XGBoost (全量训练, AM 子集评估) | 0.5590 | 0.1374 |

> **结论**: ✗ 模型 AUROC 比 AlphaMissense 低 **-0.0784** —— 模型未能打败 AlphaMissense 基线。  
> 这不是 bug，是如实记录的结论。项目需要更强的特征（如 ΔΔG）来超越这个基线。

---

## Task 2: 主表-only 干净评估（对齐历史 0.641）

**目标**: 去掉 additional_benign 负样本，只用主表数据评估，得到和早期 Phase 3 可比的干净指标。

### 代码

```python
# (数据加载和CV工具函数同上，省略)

# ===== 主表子集 =====
main_mask = source_arr == "main"
X_main = X_full[main_mask]
y_main = y_bin[main_mask]
g_main = groups[main_mask]

print(f"主表-only: n={len(y_main)}, 正例={int(y_main.sum())}, "
      f"base_rate={y_main.sum()/len(y_main):.4f}")

oof_main, folds_main = cv_evaluate_binary(X_main, y_main, g_main)
r_main = print_metrics("v3 XGBoost (主表-only)", y_main, oof_main, folds_main)

# 全量对比
oof_full, folds_full = cv_evaluate_binary(X_full, y_bin, groups)
full_auroc = roc_auc_score(y_bin, oof_full)
full_auprc = average_precision_score(y_bin, oof_full)
r_full = print_metrics("v3 XGBoost (全量)", y_bin, oof_full, folds_full)
```

### 输出

```
主表-only: n=2089, 正例=222, base_rate=0.1063

============================================================
  v3 XGBoost (主表-only)
  n=2089, pos=222, base_rate=0.1063
  AUROC = 0.5784
  AUPRC = 0.1495  (base_rate=0.1063)
  Per-fold AUROC: ['0.601', '0.603', '0.554', '0.562', '0.593']  mean=0.5826 ± 0.0207
  Per-fold AUPRC: ['0.169', '0.188', '0.129', '0.215', '0.158']  mean=0.1717 ± 0.0290

============================================================
  v3 XGBoost (全量)
  n=2179, pos=222, base_rate=0.1019
  AUROC = 0.5603
  AUPRC = 0.1308  (base_rate=0.1019)
  Per-fold AUROC: ['0.594', '0.594', '0.409', '0.631', '0.564']  mean=0.5583 ± 0.0776
  Per-fold AUPRC: ['0.133', '0.164', '0.089', '0.164', '0.153']  mean=0.1405 ± 0.0282
```

### Task 2 汇总

| 指标 | 主表-only | 全量 | 差异 |
|---|---|---|---|
| AUROC | 0.5784 | 0.5603 | +0.0181 |
| AUPRC | 0.1495 | 0.1308 | +0.0187 |
| n | 2089 | 2179 | |
| 正例 | 222 | 222 | |
| base_rate | 0.1063 | 0.1019 | |

> **结论**: ✓ 主表-only AUROC 与全量差异仅 +0.0181，additional_benign 未造成明显指标虚高。

---

## Task 3: 检查 additional_benign 是否制造"捷径"或分布漂移

**目标**: 确认那 90 个来自另一张表的负样本没有让模型学到"识别数据来源"这种假信号。

### 实验 A: 排除 additional_benign 负例

```python
# (数据加载和CV工具函数同上，省略)

# ===== 实验 A: 正例 + 主表负例 (排除 additional_benign) =====
mask_a = (source_arr == "main") | (y_bin == 1)
X_a = X_full[mask_a]
y_a = y_bin[mask_a]
g_a = groups[mask_a]

print(f"实验 A (正例+主表负例, 排除 additional_benign):")
print(f"  n={len(y_a)}, 正例={int(y_a.sum())}, base_rate={y_a.sum()/len(y_a):.4f}")
print(f"  排除的 additional_benign 负例: {(~mask_a).sum()} 行")

oof_a, folds_a = cv_evaluate_binary(X_a, y_a, g_a)
r_a = print_metrics("实验 A: 正例+主表负例", y_a, oof_a, folds_a)

oof_full, _ = cv_evaluate_binary(X_full, y_bin, groups)
full_auroc = roc_auc_score(y_bin, oof_full)
print(f"\n  全量 AUROC={full_auroc:.4f}  vs  实验A AUROC={r_a['auroc']:.4f}  "
      f"(delta={r_a['auroc']-full_auroc:+.4f})")
```

```
实验 A (正例+主表负例, 排除 additional_benign):
  n=2089, 正例=222, base_rate=0.1063
  排除的 additional_benign 负例: 90 行

============================================================
  实验 A: 正例+主表负例
  n=2089, pos=222, base_rate=0.1063
  AUROC = 0.5784
  AUPRC = 0.1495  (base_rate=0.1063)

  全量 AUROC=0.5603  vs  实验A AUROC=0.5784  (delta=+0.0181)
```

### 实验 B: 捷径探针 —— 预测数据来源

```python
# ===== 实验 B: 捷径探针 =====
y_probe = (source_arr == "additional_benign").astype(int)
n_probe_pos = int(y_probe.sum())
print(f"捷径探针标签: is_additional_benign=1 共 {n_probe_pos} 个 "
      f"(base_rate={n_probe_pos/len(y_probe):.4f})")

oof_probe, folds_probe = cv_evaluate_binary(
    X_full, y_probe, groups, sample_weight_mode="scale_pos_weight")

probe_auroc = roc_auc_score(y_probe, oof_probe)
probe_auprc = average_precision_score(y_probe, oof_probe)

print(f"\n捷径探针结果:")
print(f"  n={len(y_probe)}, positive={n_probe_pos}")
print(f"  AUROC = {probe_auroc:.4f}")
print(f"  AUPRC = {probe_auprc:.4f}")

if probe_auroc > 0.7:
    print(f"\n  ⚠️  警告: AUROC={probe_auroc:.3f} > 0.7")
    print(f"  特征能够区分数据来源 → 全量任务中存在被模型利用的捷径!")
else:
    print(f"\n  ✓  AUROC={probe_auroc:.3f} ≤ 0.7 → 捷径风险低")
```

```
捷径探针标签: is_additional_benign=1 共 90 个 (base_rate=0.0413)

捷径探针结果:
  n=2179, positive=90, base_rate=0.0413
  AUROC = 0.7432
  AUPRC = 0.1109 (base_rate=0.0413)

  ⚠️  警告: AUROC=0.743 > 0.7
  特征能够区分数据来源 → 全量任务中存在被模型利用的捷径!
  建议: 在报告里标注此风险，或只用主表数据评估。
```

### Task 3 汇总

| 实验 | AUROC | 结论 |
|---|---|---|
| 实验 A: pos + main-neg (排除 additional_benign) | 0.5784 | 与全量差异小(+0.0181) |
| 实验 B: 捷径探针 (预测数据来源) | **0.7432** | ⚠️ 超过 0.7 阈值，存在捷径风险 |

> **结论**: 虽然实验 A 显示 additional_benign 对最终指标影响不大，但捷径探针 AUROC=0.743 表明**特征能够以中等准确性区分数据来源**，模型可能在学习过程中利用了这种分布差异。建议在报告里标注此风险。

---

## Task 4: 补上 ΔΔG 与结构失稳特征

**目标**: 用 ESM2-650M 零样本方法 (Masked Marginal) 计算突变 ΔΔG。

**方法**: ΔΔG_proxy = log P(WT_AA | context) − log P(MT_AA | context)
- 正值 → 突变不利（去稳定化）；负值 → 突变有利（稳定化）

### 4a. 加载模型 + 准备 masked 序列

```python
import numpy as np, pandas as pd, re, time, os, warnings
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import EsmForMaskedLM, EsmTokenizer
warnings.filterwarnings("ignore")

BASE_PATH = "/mnt/volume6/czj/labLGN/LabLZ/"
MAX_SEQ_LEN = 1022
BATCH_SIZE = 8

df_main = pd.read_csv(BASE_PATH + "cell2024_model_single_subst.csv")
print(f"数据加载完毕: {len(df_main)} 行")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = EsmTokenizer.from_pretrained(BASE_PATH + "esm2_650M")
model = EsmForMaskedLM.from_pretrained(BASE_PATH + "esm2_650M").eval().to(device)
if device.type == "cuda":
    model = model.half()
MASK_ID = tokenizer.mask_token_id
```

```
数据加载完毕: 2179 行
Device: cuda
模型已加载 (半精度), GPU 显存: 2.7 GB
```

```python
# ===== 解析突变 =====
def parse_mutation(mut_str):
    if not isinstance(mut_str, str): return None, None, None
    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mut_str.strip())
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, None, None)

def truncate_around_pos(seq, pos):
    if len(seq) <= MAX_SEQ_LEN:
        return seq, pos - 1
    half = MAX_SEQ_LEN // 2
    end = min(len(seq), pos + half)
    start = max(0, end - MAX_SEQ_LEN)
    return seq[start:start + MAX_SEQ_LEN], pos - start - 1

df_ddg = df_main[["Gene", "Variant", "Mutation_used", "sequence"]].copy()
df_ddg["wt_aa"], df_ddg["pos"], df_ddg["mt_aa"] = zip(
    *df_ddg["Mutation_used"].apply(parse_mutation))

valid = df_ddg["wt_aa"].notna() & df_ddg["sequence"].notna()
df_ddg = df_ddg[valid].copy()

# 构建 masked 序列
masked_seqs, valid_indices = [], []
skip = {"wt_mismatch": 0, "pos_oob": 0, "ok": 0}

for idx, row in df_ddg.iterrows():
    seq, wt_aa, pos = row["sequence"], row["wt_aa"], row["pos"]
    if pos < 1 or pos > len(seq): skip["pos_oob"] += 1; continue
    if seq[pos-1] != wt_aa: skip["wt_mismatch"] += 1; continue
    trunc_seq, trunc_pos = truncate_around_pos(seq, pos)
    if trunc_pos < 0 or trunc_pos >= len(trunc_seq): skip["pos_oob"] += 1; continue
    masked = trunc_seq[:trunc_pos] + "<mask>" + trunc_seq[trunc_pos+1:]
    masked_seqs.append(masked)
    valid_indices.append(idx)
    skip["ok"] += 1

print(f"就绪序列: {len(masked_seqs)} (跳过: {skip})")
df_ddg = df_ddg.loc[valid_indices].copy()
# KEY 使用 Gene + Mutation_used（而非 Variant），因为 Mutation_used 是实际用于建模的突变
KEY_ORDER = (df_ddg["Gene"] + "||" + df_ddg["Mutation_used"]).tolist()
```

```
有效突变: 2179 / 2179
就绪序列: 2179 (跳过: {'wt_mismatch': 0, 'pos_oob': 0, 'ok': 2179})
```

### 4b. 批量推理计算 ΔΔG

```python
AA_TO_TOKEN = {
    'A':4,'R':5,'N':6,'D':7,'C':8,'Q':9,'E':10,'G':11,'H':12,'I':13,
    'L':14,'K':15,'M':16,'F':17,'P':18,'S':19,'T':20,'W':21,'Y':22,'V':23
}

ddg_scores = np.full(len(masked_seqs), np.nan, dtype=np.float32)

@torch.inference_mode()
def compute_ddg_batch(batch_seqs, batch_wt, batch_mt):
    enc = tokenizer(batch_seqs, return_tensors="pt", padding=True,
                    truncation=True, max_length=MAX_SEQ_LEN+4)
    ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
    if device.type == "cuda":
        with torch.autocast("cuda"):
            logits = model(input_ids=ids, attention_mask=attn).logits.float()
    else:
        logits = model(input_ids=ids, attention_mask=attn).logits.float()
    mask_pos = (ids == MASK_ID).nonzero(as_tuple=False)
    results = np.full(len(batch_seqs), np.nan, dtype=np.float32)
    pos_dict = {}
    for r, c in mask_pos:
        if r.item() not in pos_dict: pos_dict[r.item()] = c.item()
    for i in range(len(batch_seqs)):
        if i not in pos_dict: continue
        p = pos_dict[i]
        wt, mt = batch_wt[i], batch_mt[i]
        if wt in AA_TO_TOKEN and mt in AA_TO_TOKEN:
            results[i] = logits[i,p,AA_TO_TOKEN[wt]].item() - logits[i,p,AA_TO_TOKEN[mt]].item()
    return results

print(f"开始计算 {len(masked_seqs)} 个变体的 ΔΔG (batch_size={BATCH_SIZE}) ...")
t0 = time.time()
n_batches = (len(masked_seqs) + BATCH_SIZE - 1) // BATCH_SIZE
for i in range(0, len(masked_seqs), BATCH_SIZE):
    batch_seqs = masked_seqs[i:i+BATCH_SIZE]
    batch_wt = df_ddg["wt_aa"].iloc[i:i+BATCH_SIZE].tolist()
    batch_mt = df_ddg["mt_aa"].iloc[i:i+BATCH_SIZE].tolist()
    ddg_scores[i:i+BATCH_SIZE] = compute_ddg_batch(batch_seqs, batch_wt, batch_mt)
    bn = i // BATCH_SIZE + 1
    if bn % 50 == 0 or bn == 1 or bn == n_batches:
        elapsed = time.time() - t0
        print(f"  Batch {bn}/{n_batches} ({bn/n_batches*100:.0f}%) "
              f"耗时={elapsed:.0f}s 预计剩余={elapsed/bn*(n_batches-bn):.0f}s")

elapsed = time.time() - t0
n_valid = np.isfinite(ddg_scores).sum()
print(f"\nΔΔG 计算完成! 总耗时 {elapsed:.0f}s ({elapsed/len(masked_seqs):.2f}s/变体)")
print(f"覆盖率: {n_valid}/{len(ddg_scores)} ({n_valid/len(ddg_scores)*100:.1f}%)")

ddg_finite = ddg_scores[np.isfinite(ddg_scores)]
print(f"ΔΔG 统计: mean={ddg_finite.mean():.3f}, std={ddg_finite.std():.3f}, "
      f"min={ddg_finite.min():.3f}, max={ddg_finite.max():.3f}")

df_out = pd.DataFrame({"KEY": KEY_ORDER, "ddg_esm2": ddg_scores})
df_out.to_csv(BASE_PATH + "ddg_esm2.csv", index=False)
print("已保存 ddg_esm2.csv")
```

```
  Batch 1/273 (0%) 耗时=1s 预计剩余=327s
  ...
  Batch 273/273 (100%) 耗时=33s 预计剩余=0s

ΔΔG 计算完成! 总耗时 33s (0.02s/变体)
覆盖率: 2179/2179 (100.0%)
ΔΔG 统计: mean=-0.450, std=3.221, min=-14.918, max=13.828
已保存 ddg_esm2.csv
```

### 4c. 将 ΔΔG 加入特征集，重新二分类评估

```python
# ===== 将 ΔΔG 映射到全量 2179 行 =====
ddg_map = dict(zip(KEY_ORDER, ddg_scores))
# KEY 使用 Gene + Mutation_used，与上方 KEY_ORDER 一致
full_keys = (df_main["Gene"] + "||" + df_main["Mutation_used"]).tolist()
ddg_full = np.array([ddg_map.get(k, np.nan) for k in full_keys], dtype=np.float32)

# ===== 特征矩阵 =====
df_feat = pd.read_csv(BASE_PATH + "features_v3.csv")
ID_COLS = ["KEY", "Gene", "reloc_v3"]
NAN_PLACEHOLDERS = ["ddg", "plddt_diff"]
exclude_cols = ID_COLS + NAN_PLACEHOLDERS
feat_cols = [c for c in df_feat.columns if c not in exclude_cols]

X_base = df_feat[feat_cols].values.astype(np.float32)
X_with_ddg = np.hstack([X_base, ddg_full.reshape(-1, 1)])

y_bin = (df_feat["reloc_v3"].values > 0).astype(int)
groups = df_feat["Gene"].values

# ===== CV 对比: 有/无 ΔΔG =====
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_ddg = np.zeros(len(y_bin), dtype=np.float32)
oof_noddg = np.zeros(len(y_bin), dtype=np.float32)

for fold, (tr_idx, te_idx) in enumerate(cv.split(X_with_ddg, y_bin, groups)):
    y_tr = y_bin[tr_idx]
    # --- 有 ΔΔG ---
    imp = SimpleImputer(strategy="median"); scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(X_with_ddg[tr_idx])).astype(np.float32)
    X_te = scl.transform(imp.transform(X_with_ddg[te_idx])).astype(np.float32)
    sw = compute_sample_weight("balanced", y_tr)
    m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.5,
                      objective="binary:logistic", eval_metric="aucpr",
                      random_state=42, n_jobs=-1, tree_method="hist")
    m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    oof_ddg[te_idx] = m.predict_proba(X_te)[:, 1]

    # --- 无 ΔΔG ---
    imp_b = SimpleImputer(strategy="median"); scl_b = StandardScaler()
    X_tr_b = scl_b.fit_transform(imp_b.fit_transform(X_base[tr_idx])).astype(np.float32)
    X_te_b = scl_b.transform(imp_b.transform(X_base[te_idx])).astype(np.float32)
    sw_b = compute_sample_weight("balanced", y_tr)
    m_b = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.5,
                        objective="binary:logistic", eval_metric="aucpr",
                        random_state=42, n_jobs=-1, tree_method="hist")
    m_b.fit(X_tr_b, y_tr, sample_weight=sw_b, verbose=False)
    oof_noddg[te_idx] = m_b.predict_proba(X_te_b)[:, 1]

    y_te = y_bin[te_idx]
    print(f"  Fold {fold}: +ddg AUROC={roc_auc_score(y_te, oof_ddg[te_idx]):.4f}  "
          f"baseline AUROC={roc_auc_score(y_te, oof_noddg[te_idx]):.4f}")

auroc_ddg = roc_auc_score(y_bin, oof_ddg)
auprc_ddg = average_precision_score(y_bin, oof_ddg)
auroc_no = roc_auc_score(y_bin, oof_noddg)
auprc_no = average_precision_score(y_bin, oof_noddg)
```

```
特征矩阵: (2179, 1289)
ddg 列 NaN 数: 0/2179
  Fold 0: +ddg AUROC=0.6196  baseline AUROC=0.5943
  Fold 1: +ddg AUROC=0.6234  baseline AUROC=0.5936
  Fold 2: +ddg AUROC=0.4876  baseline AUROC=0.4091
  Fold 3: +ddg AUROC=0.6253  baseline AUROC=0.6308
  Fold 4: +ddg AUROC=0.6050  baseline AUROC=0.5636
```

### Task 4 最终结果

| 指标 | v3 基线 | v3 + ΔΔG | 增量 |
|---|---|---|---|
| **AUROC** | 0.5603 | **0.5932** | **+0.0328** |
| AUPRC | 0.1308 | 0.1435 | +0.0127 |

### 4d. 特征重要性（含 ΔΔG 的排名）

```
Top-30 特征重要性:
   1. esm_387      0.00494
   2. esm_914      0.00452
   3. esm_291      0.00439
   ... (全部 esm_*)
  30. esm_950      0.00276

  ΔΔG (ddg_esm2): 排名=244/1289, 重要性=0.00131
  → ΔΔG 排名较后，增量有限
```

> **结论 (旧版 tokenizer)**: ΔΔG 带来了 **AUROC +0.033** 的增量（0.560→0.593）。但 ΔΔG 在 1289 个特征中仅排 244 位。  
> **修正后 (正确 tokenizer AA_TO_TOKEN)**: ΔΔG **排名飙升至第 6 位**（重要性 0.00391），AUROC 增量 +0.036（0.560→0.596）。  
> **100% 覆盖率** (2179/2179)，零样本方法非常适合大规模筛选。

### 4e. ΔΔG 与 AlphaMissense 的相关性分析

```python
from scipy.stats import pearsonr, spearmanr

# ===== 计算 ddg_esm2 与 AlphaMissense 的相关性 =====
am_scores = df_main["AlphaMissense score"].values.astype(np.float64)

# 两者都有效的行
mask_corr = np.isfinite(ddg_full) & np.isfinite(am_scores)
n_corr = mask_corr.sum()
print(f"有效样本 (两者均非NaN): {n_corr}/{len(ddg_full)}")

# Pearson 相关系数
r_pearson, p_pearson = pearsonr(ddg_full[mask_corr], am_scores[mask_corr])
# Spearman 秩相关系数
r_spearman, p_spearman = spearmanr(ddg_full[mask_corr], am_scores[mask_corr])

print(f"\nddg_esm2 vs AlphaMissense score 相关性:")
print(f"  Pearson  r = {r_pearson:+.4f}  (p = {p_pearson:.2e})")
print(f"  Spearman r = {r_spearman:+.4f}  (p = {p_spearman:.2e})")

# 按标签分别看
print(f"\n按标签分组:")
for label, name in [(0, "负例 (不重定位)"), (1, "正例 (重定位)")]:
    mask_label = mask_corr & (y_bin == label)
    n_label = mask_label.sum()
    if n_label > 5:
        rp, pp = pearsonr(ddg_full[mask_label], am_scores[mask_label])
        rs, ps = spearmanr(ddg_full[mask_label], am_scores[mask_label])
        print(f"  {name:20s} (n={n_label:4d}): Pearson r={rp:+.4f}, Spearman r={rs:+.4f}")

# 解读
abs_r = abs(r_spearman)
if abs_r < 0.3:
    print(f"  |r| = {abs_r:.3f} < 0.3 → ΔΔG 与 AlphaMissense 几乎正交")
    print(f"  → 两者捕捉的几乎是完全不同的信号，组合使用价值高")
elif abs_r < 0.6:
    print(f"  0.3 ≤ |r| = {abs_r:.3f} < 0.6 → 中等相关")
else:
    print(f"  |r| = {abs_r:.3f} ≥ 0.6 → 较强相关")
```

```
(运行后填入实际数值)
有效样本 (两者均非NaN): 2053/2179

ddg_esm2 vs AlphaMissense score 相关性:
  Pearson  r = X.XXXX  (p = X.XXe-XX)
  Spearman r = X.XXXX  (p = X.XXe-XX)

按标签分组:
  负例 (不重定位) (n=1832): Pearson r=X.XXXX, Spearman r=X.XXXX
  正例 (重定位)   (n= 221): Pearson r=X.XXXX, Spearman r=X.XXXX

  |r| = X.XXX → (解读)
```

> **关键问题**: ddg_esm2 和 AlphaMissense 是正交信号还是冗余信号？相关性结果决定两者能否互补。

---

## Task 5: PCA 压缩 ESM2，让结构特征浮出来

**目标**: 1280 维 ESM2 把结构特征完全淹没。降维后看结构特征能否进入重要位次。

### 代码

```python
import numpy as np, pandas as pd, warnings
from sklearn.decomposition import PCA
# (其他 imports 同上)

# 识别 ESM2 列和结构列
esm_cols = [c for c in feat_cols if c.startswith("esm_")]
struct_cols_present = ["plddt", "sasa", "rsa", "ss_helix", "ss_strand",
                       "ss_coil", "delta_hydrophobicity", "struct_status"]

X_esm = X_full[:, esm_idx]
X_struct = X_full[:, struct_idx]

pca_results = {}
for n_comp in [30, 50, 100]:
    oof_pca = np.zeros(len(y_bin), dtype=np.float32)
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_full, y_bin, groups)):
        # ESM2: Impute → Scale → PCA (仅在训练集 fit)
        imp_e = SimpleImputer(strategy="median"); scl_e = StandardScaler()
        Xe_tr = scl_e.fit_transform(imp_e.fit_transform(X_esm[tr_idx])).astype(np.float32)
        Xe_te = scl_e.transform(imp_e.transform(X_esm[te_idx])).astype(np.float32)
        pca = PCA(n_components=n_comp, random_state=42)
        Xe_tr_pca = pca.fit_transform(Xe_tr).astype(np.float32)
        Xe_te_pca = pca.transform(Xe_te).astype(np.float32)

        # 结构特征: 只做 Impute + Scale，不参与 PCA
        imp_s = SimpleImputer(strategy="median"); scl_s = StandardScaler()
        Xs_tr = scl_s.fit_transform(imp_s.fit_transform(X_struct[tr_idx])).astype(np.float32)
        Xs_te = scl_s.transform(imp_s.transform(X_struct[te_idx])).astype(np.float32)

        # 拼接
        X_tr_pca = np.hstack([Xe_tr_pca, Xs_tr])
        X_te_pca = np.hstack([Xe_te_pca, Xs_te])

        y_tr = y_bin[tr_idx]; sw = compute_sample_weight("balanced", y_tr)
        model = XGBClassifier(...)
        model.fit(X_tr_pca, y_tr, sample_weight=sw, verbose=False)
        oof_pca[te_idx] = model.predict_proba(X_te_pca)[:, 1]

    auroc = roc_auc_score(y_bin, oof_pca)
    auprc = average_precision_score(y_bin, oof_pca)
    pca_results[n_comp] = {"auroc": auroc, "auprc": auprc, ...}
```

### 5a. PCA 维度扫描汇总

| PCA 维度 | AUROC | AUPRC | 解释方差 |
|---|---|---|---|
| Full (1280) | 0.5603 | 0.1308 | 1.00 |
| **PCA(30)** | **0.6063** | 0.1381 | 0.49 |
| **PCA(50)** | **0.6087** | 0.1388 | 0.58 |
| PCA(100) | 0.5752 | 0.1343 | 0.70 |

> **PCA(50) 达到最高 AUROC=0.6087**，比全量 ESM2 (0.5603) 提升 +0.0484！

### 5b. PCA(50) 后的特征重要性

```
Top-20 特征重要性 (PCA(50) 后):
   1. PC1                 0.03355
   2. PC36                0.02584
   3. PC33                0.02287
   4. PC30                0.02216
   5. PC23                0.02203
   6. ss_helix            0.02114  ★ 结构特征进入 top-20!
   ...
  16. delta_hydrophobicity 0.01907 ★ 结构特征进入 top-20!

结构特征在 PCA(50) 后的排名:
  plddt                   排名= 30/58  重要性=0.01664
  sasa                    排名= 21/58  重要性=0.01821
  rsa                     排名= 23/58  重要性=0.01793
  ss_helix                排名=  6/58  重要性=0.02114 ★
  ss_strand               排名= 57/58  重要性=0.00957
  ss_coil                 排名= 38/58  重要性=0.01616
  delta_hydrophobicity    排名= 16/58  重要性=0.01907 ★
  struct_status           排名= 58/58  重要性=0.00000
```

> **结论**: PCA 降维效果显著！
> - AUROC 从 0.560 提升到 **0.609** (+0.048)，超过所有其他改进
> - **ss_helix**(排名 6) 和 **delta_hydrophobicity**(排名 16) 成功进入 top-20
> - PCA(50) 仅用 50 维保留了 58% 的方差，大幅减少了 ESM2 的噪声维度
> - PCA(30) 也达到 0.606，说明 1280 维 ESM2 中大部分是冗余信息

---

## Task 6: 多分类（5 类去向）

**目标**: 预测蛋白质错定位的具体去向。

### 标签分布

| Class | 描述 | n |
|---|---|---|
| C0 | 不重定位 | 1957 |
| C1 | 聚集/C2 | 34 |
| C2 | 分泌途径/C3 | 121 |
| C3 | 核定位/C4 | 29 |
| C4 | 细胞质/C5 | 38 |

### 代码

```python
import numpy as np, pandas as pd, warnings
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
# (其他 imports 同上)

cv_multi = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_5 = np.zeros((len(y_5class), 5), dtype=np.float32)

for fold, (tr_idx, te_idx) in enumerate(cv_multi.split(X_full, y_5class, groups)):
    imp = SimpleImputer(strategy="median"); scl = StandardScaler()
    X_tr = scl.fit_transform(imp.fit_transform(X_full[tr_idx])).astype(np.float32)
    X_te = scl.transform(imp.transform(X_full[te_idx])).astype(np.float32)
    sw = compute_sample_weight("balanced", y_5class[tr_idx])
    model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          objective="multi:softprob", eval_metric="mlogloss",
                          random_state=42, n_jobs=-1, tree_method="hist")
    model.fit(X_tr, y_5class[tr_idx], sample_weight=sw, verbose=False)
    oof_5[te_idx] = model.predict_proba(X_te)
```

### 输出

```
  Fold 0: acc=0.911, class分布={0:388, 1:4, 2:25, 3:5, 4:4}
  Fold 1: acc=0.909, class分布={0:388, 1:6, 2:15, 3:5, 4:13}
  Fold 2: acc=0.850, class分布={0:403, 1:7, 2:46, 3:7, 4:11}
  Fold 3: acc=0.919, class分布={0:388, 1:7, 2:17, 3:5, 4:5}
  Fold 4: acc=0.909, class分布={0:390, 1:10, 2:18, 3:7, 4:5}
```

### Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 不重定位(C0) | 0.899 | **1.000** | 0.947 | 1957 |
| 聚集/C2 | **1.000** | 0.059 | 0.111 | 34 |
| 分泌途径/C3 | 0.000 | 0.000 | 0.000 | 121 |
| 核定位/C4 | 0.000 | 0.000 | 0.000 | 29 |
| 细胞质/C5 | 0.000 | 0.000 | 0.000 | 38 |

- **Accuracy**: 0.8986
- **Macro F1**: 0.2115
- **Weighted F1**: 0.8518

### Confusion Matrix

| 真实\预测 | C0 | C2 | C3 | C4 | C5 |
|---|---|---|---|---|---|
| C0 (n=1957) | 1956 | 0 | 1 | 0 | 0 |
| C2 (n=34) | 32 | 2 | 0 | 0 | 0 |
| C3 (n=121) | 121 | 0 | 0 | 0 | 0 |
| C4 (n=29) | 29 | 0 | 0 | 0 | 0 |
| C5 (n=38) | 38 | 0 | 0 | 0 | 0 |

### Per-Class AUROC (One-vs-Rest)

| Class | AUROC | Support |
|---|---|---|
| C0 (不重定位) | 0.550 | 1957 |
| C1 (聚集/C2) | 0.580 | 34 |
| C2 (分泌途径/C3) | 0.566 | 121 |
| C3 (核定位/C4) | 0.563 | 29 |
| C4 (细胞质/C5) | 0.553 | 38 |

> **结论**: 多分类效果很差。
> - 模型几乎把所有样本都预测为 C0（不重定位），Accuracy=0.899 看似高，但这只是因为 C0 占了 90%
> - **C3(分泌途径, n=121) 没有任何一个被正确预测** — precision/recall 全部为 0
> - 仅 C2(聚集) 有 2/34 被正确识别 (precision=1.0 因为只有 2 个被预测为 C2)
> - 所有 per-class AUROC 都在 0.55-0.58 之间，接近随机
> - **结论: 当前特征集无法区分具体的错定位去向**，5 分类目前没有信号

---

## 最终汇总

### 主表

| 实验名 | 评估子集 | n | 正例 | base_rate | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| AlphaMissense-alone | AM子集 (n=2053) | 2053 | 221 | 0.1076 | **0.6374** | 0.1547 |
| v3 XGBoost (AM CV) | AM子集 | 2053 | 221 | 0.1076 | 0.5719 | 0.1527 |
| v3 XGBoost (full→AM) | AM子集 | 2053 | 221 | 0.1076 | 0.5590 | 0.1374 |
| v3 XGBoost (主表-only) | 主表 | 2089 | 222 | 0.1063 | 0.5784 | 0.1495 |
| v3 XGBoost (全量基线) | 全量 | 2179 | 222 | 0.1019 | 0.5603 | 0.1308 |
| v3 + ΔΔG (ESM2) | 全量 | 2179 | 222 | 0.1019 | **0.5932** | 0.1435 |
| v3 + PCA(30) | 全量 | 2179 | 222 | 0.1019 | 0.6063 | 0.1381 |
| **v3 + PCA(50)** | 全量 | 2179 | 222 | 0.1019 | **0.6087** | 0.1388 |
| v3 + PCA(100) | 全量 | 2179 | 222 | 0.1019 | 0.5752 | 0.1343 |
| 多分类 (5-class) | 全量 | 2179 | 222 | 0.1019 | macro-F1=0.2115 | weighted-F1=0.8518 |

### 捷径探针

| 实验 | AUROC | 结论 |
|---|---|---|
| Exp A: pos + main-neg | 0.5784 | 与全量差异小 (+0.018) |
| Exp B: 来源探针 | **0.7432** | ⚠️ 特征可区分数据来源 |

---

## 核心问题回答

### Q1: 自建模型能否 ≥ AlphaMissense-alone？

**不能。** 在 AM 子集（n=2053，行完全相同）上：
- AlphaMissense-alone AUROC = **0.6374**
- v3 XGBoost (full→AM) = 0.5590
- 差距 = **−0.0784**

即使用了 ΔΔG (+0.033) 和 PCA (+0.048)，最佳模型 AUROC 也仅到 0.609，仍低于 AlphaMissense 的 0.637。

### Q2: ΔΔG 是否带来可见增量？

**是，但有限。**
- v3 基线: 0.5603
- v3 + ΔΔG: **0.5932 (+0.033)**
- 这是所有单项改进中最大的提升
- 但 ΔΔG 在 1289 个特征中仅排 244 位，仍被 ESM2 维度淹没
- 覆盖率 100% (2179/2179)，零样本方法可行

### Q3: PCA 是否让结构特征浮出来？

**是，且提升最大。**
- PCA(50) AUROC = **0.6087**，比全量 ESM2 提升 **+0.048**
- ss_helix (排名 6) 和 delta_hydrophobicity (排名 16) 进入 top-20
- 1280 维 ESM2 中大部分是冗余信息，仅需 30-50 维即可超越全量

### Q4: 多分类是否有信号？

**几乎没有。**
- 模型几乎把所有样本预测为 C0（不重定位）
- C3 (分泌途径, n=121) precision/recall 全部为 0
- 所有 per-class AUROC 在 0.55-0.58 之间
- 当前特征集无法区分具体的错定位去向

### Q5: additional_benign 是否有捷径风险？

**有风险。** 捷径探针 AUROC = 0.743 (>0.7)，特征能够以中等准确性区分数据来源。虽然实验 A 显示对整体指标影响不大 (+0.018)，但建议在报告里标注此风险。

---

## 关键发现总结

1. **AlphaMissense 是最强基线** (AUROC=0.6374)，当前 XGBoost 模型无法超越
2. **PCA 降维是最大的单项改进** (+0.048 AUROC)，说明 ESM2 1280 维高度冗余
3. **ΔΔG 提供了真实增量** (+0.033 AUROC)，覆盖 100%，但排名不高
4. **结构特征在 PCA 后成功浮出** (ss_helix 排名第 6)
5. **多分类无信号**：模型完全无法区分具体的错定位去向
6. **additional_benign 存在捷径风险** (AUROC=0.743)，需在报告中标注
