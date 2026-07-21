# PUPS LABCODE 项目文档

> 2026-07-13 | conda: `localization` | PyTorch 2.6.0+cu124 | 2× RTX 4090

---

## 架构总览

```
cell2024_model_single_subst.csv (2179变体)
    │
    ├─→ make_gating_csv.py        → gating_variants.csv (25变体, 快速测试)
    │       └─→ run_gating_batch.py → gating_results.csv
    │
    ├─→ run_pups_full_A.ipynb    → pups_full_delta.pkl  {KEY: float32[29]}
    │       └─→ run_pups_full_A_v2.ipynb (v2标签版)
    │
    └─→ test_incremental.ipynb   → AUROC/AUPRC (PUPS vs ESM2 增量)
            └─→ test_incremental_v2.ipynb (v2标签版)

共同依赖: pups_inference.py ─→ PUPS/src/ (SubCellProtModel)
```

---

## 文件说明

### 核心引擎: `pups_inference.py`

**作用**: ESM2 编码 + PUPS 模型加载 + 推理，被所有脚本 import。

**关键常数**: `MAX_SEQ_LEN=2000` (与PUPS训练一致, 覆盖99.6%序列), `ESM_LAYER=33`, `EMB_DIM=1280`, `CLASSES`=29个区室列表.

**esm2_encode(seq, pos=None)**: 序列→ESM2-650M每残基hidden states `(1,1,L+2,1280)`。长序列时围绕`pos`取窗口防止突变位点被截。已验证Meta `esm`库与HF `transformers`输出完全一致(max diff=0.0)，无需切换库。

**load_model()**: 加载PUPS checkpoint → `SubCellProtModel`.eval()

**模型内部数据流**:
```
序列 → [ESM2-650M] → (L+2,1280) → [LightAttentionNN] → 300维蛋白特征
                                                              ├→ [MLP] → 29维概率 ← 我们要的
                                                              └→ [U-Net]+landmark → 抗体染色图 ← 不用
```
多标签预测只依赖蛋白特征，不依赖landmark。固定landmark不影响29维输出。

**predict(model, seq, landmark, pos=None)**: 推理入口，返回`image(128,128)` + `multilabel(29,)` sigmoid概率。

**delta_report()**: 打印WT vs MT变化最大的topk区室。使用L1范数(`sum|Δ|`)，因为L1="总概率偏移量"，直觉清晰。

**L1 vs L2**: L1线性累加所有区室变化, L2平方放大单区室大幅变化。此任务用L1。在LR中使用完整29维向量时范数选择不影响结果。

### 其他文件

| 文件 | 用途 |
|------|------|
| `fetch_hpa_test.py` | 从HPA下载一张IF图片为landmark `[3,128,128]` (blue/red/yellow → 核/微管/第三通道) |
| `real_landmark.npy` | 下载的固定landmark, 所有变体共用 |
| `make_gating_csv.py` | 按5类均衡抽样25变体, 输出`gating_variants.csv` |
| `run_gating_test.py` | 单变体冒烟测试 (硬编码UTRN S25R) |
| `run_gating_batch.py` | 25变体批量PUPS → `gating_results.csv`, 计算|Δ|L1 vs reloc的AUROC |
| `run_pups_full_A.ipynb` | 全量2179变体PUPS delta生成, 每20个增量保存 |
| `run_pups_full_A_v2.ipynb` | v2版: C1与Mislocalized=0合并 |
| `test_incremental.ipynb` | PUPS(29维) vs ESM2(1280维)增量: StratifiedGroupKFold+LR |
| `test_incremental_v2.ipynb` | v2版: 使用reloc_v2标签 |
| `smoke_test.py` | 最小连通性测试(随机landmark) |

---

## 数据格式

### cell2024_model_single_subst.csv (2179行)

| 列 | 含义 | 用于建模 |
|----|------|---------|
| Gene, Variant, Mutation_used | 标识 | 分组(Gene) + KEY |
| sequence, mutant_sequence | WT/MT全长序列 | ESM2编码 |
| Mislocalized | 原论文二值标签 (0/1) | **v2基础标签** |
| label_5class | C1~C5 细分类 (仅235行有值) | 细分类标签 |
| plddt, sasa, rsa | AlphaFold结构特征 | **XGBoost特征** |
| ss_type, delta_hydrophobicity | 二级结构+疏水性变化 | **XGBoost特征** |

### 标签交叉表 (Mislocalized × label_5class)

| Mislocalized | C1_no_reloc | C2 | C3 | C4 | C5 | NaN | 合计 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0 | 0 | 0 | 0 | 0 | **1943** | 1943 |
| 1.0 | **13** | 34 | 121 | 29 | 38 | 1 | 236 |

- C1_no_reloc: 表型描述为"X>X"(同区室), 但原论文标为Mislocalized=1 **(矛盾)** → v2归入不重定位
- 1个NaN: **RPE65 K294T**, 来自additional_benign表, 无表型描述

### Pickle文件格式

```
pups_full_delta.pkl        → dict{"Gene||Variant": float32[29]}  (delta = P_mt - P_wt)
phase4_esm2_local_delta.pkl → dict{"Gene||Variant": float32[1280]} (ESM2局部窗口delta)
```

---

## C1_no_reloc 详解

13个C1变体的表型格式为`WT位置>MT位置`，全部显示"同区室":

| Gene | Mutation | Phenotype |
|------|----------|-----------|
| ACTB, ACTN4 | R183W, K255E | Actin>Actin |
| CYP1A2, FGFR1, GGCX×2, IL12RB1, KEL, SLC39A4, SLC7A9 | 各 | ER>ER |
| FECH, SDHD, TSFM | 各 | Mitochondria>Mitochondria |

**"ER>ER"字面意思是"从ER到ER"——没有重定位。** 原论文标为Mislocalized=1可能是因为实验筛选中检测到表达量/折叠异常，但对"预测重定位到哪"的任务，这些应归入不重定位类。

---

## 实验结果

### 修复记录

| 问题 | 修复 |
|------|------|
| PyTorch CPU-only (pups env) | 改用`localization` env (PyTorch 2.6.0+cu124) |
| `r.gene` 列名不匹配 | 改为`r.Gene` |
| `esm2_encode`长序列从头截断 | 新增`pos`参数, 围绕突变位点取窗口 |
| `load_models`函数不存在 | 改为`load_model` |
| Cyrillic占位符`MKTАYIAKQR` | 改为纯ASCII |
| `fsspec`缺失 | `pip install fsspec` |

### 门控测试 (n=25)

| 分组 | 数量 | 平均|Δ| L1 |
|------|------|------|
| C4_nuclear | 4 | 0.060 |
| C5_cytoplasmic | 4 | 0.057 |
| C3_secretory | 4 | 0.053 |
| C2_aggregation | 4 | 0.043 |
| C1_no_reloc | 9 | 0.038 |
| 重定位(C2-C5) | 16 | 0.053 |
| 不重定位(C1) | 9 | 0.038 |

AUROC(|Δ|→reloc) = **0.646** (n=25, 不可靠)

### v1 vs v2 对比

```
v1标签: reloc = (label_5class ≠ C1)          仅235样本
v2标签: reloc = Mislocalized; C1强制=0      全部2179样本
```

| 判据 | v1 | v2 |
|------|----|----|
| 样本数 (不重/重) | 235 (13/222) | **2179 (1956/223)** |
| **判据②**: L1(|Δ|)→reloc AUROC | 0.553 | **0.582** |
| **判据③**: PUPS only AUROC (AUPRC) | 0.526 | 0.515 (0.128) |
| **判据③**: ESM2 only AUROC (AUPRC) | 0.516 | **0.533** (0.121) |
| **判据③**: Combined AUROC (AUPRC) | 0.526 | **0.543** (0.126) |
| **判据③**: 增量 Δ | +0.010 | +0.010 |

**判据②**: 单变量|Δ| L1排序能力。v2因n增大更可靠(0.582)。

**判据③**: 5-fold StratifiedGroupKFold (按Gene分组) + LR + 完整特征向量。AUPRC更关注少数类。

### 核心结论

- **v2标签更合理**: 1956不重定位 vs 223重定位, 使用全部2179样本
- **ESM2有微弱信号**: AUROC=0.533 > 0.5, AUPRC=0.121 (随机≈0.10)
- **PUPS无增量**: Δ=+0.010, 29维定位概率对单氨基酸替换不敏感
- **序列嵌入不够**: 需加入结构特征(pLDDT/SASA/二级结构/疏水性)

---

## 分类方案

### 方案 A: 二分类 (推荐起步)

```
Class 0 (不重定位): Mislocalized=0 (1943) + C1 (13) = 1956
Class 1 (重定位):   C2+C3+C4+C5 (222) + RPE65 K294T (1) = 223
正负比 1:8.8 → XGBoost + scale_pos_weight
特征: ESM2(1280) + PUPS(29) + 结构特征(plddt/sasa/rsa/ss/delta_hydro)
评估: StratifiedGroupKFold(groups=Gene), AUROC + AUPRC
```

### 方案 B: 3分类 (折中)

```
Class 0: 不重定位 (1956)
Class 1: 分泌途径 C3 (121) — ER滞留是最常见错位机制
Class 2: 其他重定位 C2+C4+C5 (101) — 聚集+核+细胞质
```

### 方案 C: 5分类 (不推荐)

C4仅29例, 样本太少。

**建议路径**: 方案A二分类XGBoost → 如果AUPRC>0.3, 再试方案B三分类。

---

## v3 XGBoost 实验结果

> 2026-07-13 | `4.3_build_features_v3.ipynb` + `4.3_train_xgboost_v3.ipynb` | 1288特征 | conda localization

### 特征矩阵

| 特征组 | 维度 | 说明 |
|--------|------|------|
| ESM2 local delta | 1280 | WT vs MT 局部窗口嵌入差值 |
| 结构特征 | 8 | plddt, sasa, rsa, ss_helix, ss_strand, ss_coil, delta_hydrophobicity, struct_status |
| 占位 (待算) | 2 | ΔΔG, pLDDT_diff (当前全为NaN) |
| **合计** | **1288** | |

> 已移除: PUPS delta (29维, 无增量)、AlphaMissense score (属于外部预测器, 不作为特征)、HeLa TPM

### 标签 (v3 多分类)

| Class | 含义 | 数量 |
|-------|------|------|
| 0 | 不重定位 (Mislocalized=0 + C1) | 1957 |
| 1 | 聚集 (C2) | 34 |
| 2 | 分泌途径 (C3) | 121 |
| 3 | 核定位 (C4) | 29 |
| 4 | 细胞质 (C5) | 38 |

### 二分类结果 (reloc vs no-reloc)

**5-fold StratifiedGroupKFold (groups=Gene) + XGBoost**

| 指标 | LR baseline (v2) | XGBoost v3 |
|------|-----------------|------------|
| AUROC | 0.533 | **0.560** |
| AUPRC | 0.121 | **0.131** |

- XGBoost 相比 LR 有微弱提升 (+0.027 AUROC, +0.010 AUPRC)
- 提升远小于预期, 说明线性模型已能捕捉大部分信号, 非线性交互有限
- AUPRC=0.131 仍接近随机基线(~0.10), 仅靠 ESM2+结构特征不足以解决此问题

### 多分类结果 (5-class)

多分类 cell 尚未运行 (输出为空)。

### Top-30 特征重要性 (二分类模型, fit on all data)

| 排名 | 特征 | 重要性 |
|------|------|--------|
| 1 | esm_1217 | 0.00559 |
| 2 | esm_273 | 0.00511 |
| 3 | esm_1031 | 0.00438 |
| 4-12 | esm_* (各维度) | 0.0031~0.0036 |
| 13 | esm_204 | 0.00299 |
| 14-30 | esm_* (各维度) | 0.0025~0.0029 |

关键发现:
- **特征重要性极平坦** (0.0056→0.0025): 1280维 ESM2 平分信息, 无主导特征
- **结构特征未进入 top-30**: plddt/sasa/rsa/ss/delta_hydro 等被高维 ESM2 完全淹没
- 需要 PCA 降维 ESM2 或使用更结构化的特征工程

### 文件

| 文件 | 用途 |
|------|------|
| `4.3_build_features_v3.ipynb` | 构建v3特征矩阵 (无PUPS/AlphaMissense/TPM) |
| `4.3_train_xgboost_v3.ipynb` | 二分类 (Cell 3) + 多分类 (Cell 4) + 特征重要性 (Cell 5) |
| `features_v3.csv` | 2179×1288 特征矩阵 |

### 与上一版的对比

| 特征配置 | AUROC | AUPRC | 说明 |
|----------|-------|-------|------|
| ESM2 + PUPS + 结构 + **AlphaMissense** + TPM | **0.603** | **0.154** | AlphaMissense 驱动大部分提升 |
| ESM2 + 结构 | 0.560 | 0.131 | 纯净特征, 无外部预测器 |

### 下一步

1. PCA 降维 ESM2 1280→50~100, 让结构特征有机会发挥作用
2. 计算 ΔΔG 和 pLDDT_diff 填补占位列
3. 多分类需要运行 Cell 4
4. 数据增强: 对正类(C1-C5)做 SMOTE 或过采样

---

## 相关文件路径

| 路径 | 内容 |
|------|------|
| `/mnt/volume6/czj/labLGN/LabLZ/` | 数据CSV + ESM2模型 + 预处理notebook |
| `/mnt/volume6/czj/PUPS/LABCODE/` | PUPS推理代码 |
| `/mnt/volume6/czj/PUPS/src/` | PUPS源码 |
| `/mnt/volume6/czj/PUPS/checkpoints/` | PUPS预训练权重 |
