# Data Preparation Pipeline

> 蛋白质错定位预测项目 —— 数据清洗、特征工程、基线评估阶段

---

## 1. 数据预处理 (`1_data_preprocessing.ipynb`)

### 数据来源

从 `cell2024_model.xlsx` 三张 Sheet 汇入：

| Sheet | 内容 | 行数 |
|---|---|---|
| Variant annotation | 主要注释表 | 3,448 → 2,280 (去NA) |
| Additional benign variants | 额外良性变异 | 95 |
| Localization screen results | 定位筛选表 | 与主表合并 |

### 关键处理步骤

1. **主表过滤**: 去除关键列 NA 后保留 **2,280** 行，其中 `Mislocalized=1` 有 250 例。
2. **UniProt 覆盖**: 2,280 行中有 111 行缺失 UniProt accession。
3. **合并 Additional benign**: 95 行额外良性变异（94 负例 + 1 正例），与主表无重叠。合并后 **2,375** 行（2,124 负 + 251 正）。
4. **突变解析**: 从 Variant 列提取 `wt_aa + pos + mt_aa`，成功率 2,338/2,375（98.4%）。
5. **缺失填充**: 2 行缺 `wt_primary`，最终输出 2,375 行 × 14 列。

**输出文件**: `cell2024_model_single_subst.csv`

---

## 2. UniProt 序列获取 (`2_uniprot.ipynb`)

### 目标

获取每个蛋白的 canonical 序列，并生成突变后的全长序列（供 ESM2 推理）。

### 执行

- **唯一 UniProt accession**: 896 个
- **REST API 拉取**: 895 成功，1 失败（含 67 个融合基因无法解析）
- **Gene→ID 补充**: 84 个缺失 UniProt 的基因通过名称映射，56 成功补回
- **最终序列覆盖**: 914 个蛋白拥有序列

### 突变序列生成

- 有序列的行: 2,307，无序列: 68
- 排除 10 个多突变 → 2,297 个单突变
- 成功生成: **2,179** 条突变序列（118 条失败，多为序列不匹配）
- 2,179 中: 1,943 负例（不重定位），236 正例（重定位）

**输出**: `cell2024_final.csv` — 2,179 行，含 WT 和突变全长序列

---

## 3. 基线评估 (`3_baseline.ipynb`)

### ESM2 delta embedding 基线

使用 ESM2-650M 计算 WT vs 突变序列的 embedding 差异（`phase3_esm2_scores.csv`），在 AlphaMissense 有效子集上评估。

**设备**: CUDA (GPU)

### 结果（n=2,053, 正例 234, 负例 1,819）

| 方法 | AUROC | AUPRC |
|---|---|---|
| **AlphaMissense** | **0.6362** | 0.1622 |
| ESM2 delta embedding | 0.5602 | 0.1475 |
| Random baseline | 0.5094 ± 0.0229 | — |

**关键结论**: ESM2 原始 embedding 差异仅为随机水平以上约 0.05 AUROC，远不及 AlphaMissense。需要更强的特征工程。

---

## 4. 分类标签 (`3.5.1_classifier_label.ipynb`)

### 5 分类体系

从 `Mislocalization phenotype` 列（如 "Plasma membrane>ER"）归纳为 5 个去向类别：

| 标签 | 含义 | 样本数 |
|---|---|---|
| `C1_no_reloc` | 同区室（无功能意义的重定位） | 13 |
| `C2_aggregation` | 聚集 | 34 |
| `C3_secretory` | 分泌途径（ER/Golgi/Vesicles/PM） | **121** |
| `C4_nuclear` | 核定位 | 29 |
| `C5_cytoplasmic` | 细胞质定位 | 38 |

- 二分类正例总计: **235**（C1-C5 均视为"重定位"）
- 1 行无标签信息，可用行: 2,178

**输出**: `cell2024_final_with_labels.csv`

---

## 5. AlphaFold PDB 与结构特征 (`3.5.2_alphafold_download.ipynb`)

### PDB 下载

- 871 个唯一蛋白的 AlphaFold PDB 下载完成（5 个失败）
- 每蛋白计算 7 个结构特征:
  - `plddt` — AlphaFold 置信度
  - `sasa` / `rsa` — 溶剂可及表面积（绝对/相对）
  - `ss_helix` / `ss_strand` / `ss_coil` — DSSP 二级结构
  - `delta_hydrophobicity` — Kyte-Doolittle 疏水性变化（WT→MT）

### 结构特征覆盖率

| 特征 | 有效样本 | 关键统计量 |
|---|---|---|
| plddt | 2,168 | min=24.28, median=94.62, max=98.94 |
| sasa | 2,168 | min=0.00, median=41.56, max=245.85 |
| rsa | 2,168 | min=0.00, median=0.22, max=0.92 |
| delta_hydrophobicity | 2,179 | min=-9.00, median=0.00, max=9.00 |
| struct_status | 2,168 ok / 11 no_pdb | **99.5% 常数，已作为无用特征移除** |

### 最终单替换训练集

- **2,179** 行，1,943 负例 / 236 正例
- base_rate = 10.78%
- scale_pos_weight ≈ 8.23

**输出**: 合并后的 `cell2024_model_single_subst.csv`（含结构特征列）

---

## 6. ESM2 局部 Delta 嵌入 (`4.0_esm2_local_delta.ipynb`)

### 方法

不取全序列 embedding 差异，改为在突变位点附近的局部窗口内提取残基级 embedding，再做差值。生成 **1,280 维** 特征向量。

### PCA 降维扫描

在 5 折 CV 下扫描 PCA 维度:

| n_components | CV AUROC |
|---|---|
| 2 | 0.568 |
| 5 | 0.553 |
| 10 | 0.566 |
| 20 | 0.571 |
| 30 | 0.574 |
| 50 | 0.571 |
| 75 | 0.592 |
| **100** | **0.594** |
| 150 | 0.578 |

**结论**: PCA(100) 最优 (AUROC=0.594)，但后续实验发现 PCA(50) 在加入结构特征后更稳。

**输出**: 2,179 × 1,280 的 ESM2 局部 delta embedding 矩阵（已合并入 `features_v3.csv`）

---

## 数据准备阶段总结

| 指标 | 值 |
|---|---|
| 最终样本数 | **2,179** |
| 正例 (重定位) | **235** (10.78%) |
| 负例 (不重定位) | 1,944 |
| 唯一基因 | 871 |
| 唯一 UniProt | 871 |
| 有 PDB 结构 | 2,168 (99.5%) |
| ESM2 特征维度 | 1,280 |
| 结构特征维度 | 7 |
| AlphaMissense 基线 AUROC | **0.6362** |
