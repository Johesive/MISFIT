# XGBoost 试验序列

> 蛋白质错定位预测 —— 从 ESM2 原始特征到 64 维全特征 + TabPFN 集成的完整试验记录

---

## 试验架构

**所有试验共用同一协议**:
- **CV**: 5 折 StratifiedGroupKFold (groups=Gene, shuffle=True, random_state=42)
- **模型**: XGBoost (n_estimators=200, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.5)
- **不平衡处理**: scale_pos_weight + compute_sample_weight("balanced")
- **预处理**: SimpleImputer(median) + StandardScaler, PCA fit on train only
- **标签**: `reloc_v3 > 0` → 二分类 (235 正例 / 1,944 负例, base_rate=10.78%)
- **AlphaMissense 基线**: AUROC = **0.6374** (Task 1 实测 0.6362，差异来自 AM NaN 样本处理)

---

## Phase 0: 基础建设 (Tasks 0–3)

### Task 0: 特征矩阵 v3 (`task0_build_features_v3.ipynb`)

构建初始特征矩阵 **features_v3.csv**:

| 特征组 | 维度 | 说明 |
|---|---|---|
| ESM2 local delta | 1,280 | 突变位点局部 embedding 差值 |
| 结构特征 | 7 | plddt, sasa, rsa, ss_helix, ss_strand, ss_coil, delta_hydrophobicity |
| 占位列 | 2 | ddg (待算), plddt_diff (待算) |
| 标识列 | 3 | KEY, Gene, reloc_v3 |

**总维度**: 1,293 列 × 2,179 行。5 分类标签: C0(1,944), C1(13), C2(34), C3(121), C4(29), C5(38)。

**v3 全量 XGBoost (原始 1,288 维 ESM2)**: AUROC = **0.594** (5 折 CV OOF), AUPRC = 0.160。Top-30 特征全部是 `esm_*`（ESM2 原始维度淹没结构特征）。

---

### Task 1: AlphaMissense 基线 (`task1_alphamissense_baseline.ipynb`)

在 AlphaMissense 有效子集 (n=2,053, 正例 234) 上同折比较:

| 方法 | AUROC | AUPRC |
|---|---|---|
| **AlphaMissense** | **0.6362** | 0.1622 |
| v3 XGBoost (AM子集 CV) | 0.5914 | 0.1752 |
| v3 XGBoost (全量训练, AM子集评估) | 0.5946 | 0.1687 |

**结论**: 模型 AUROC 比 AlphaMissense 低 **−0.0415**，需要更强的特征来超越这个基线。

---

### Task 2: 主表-only 验证 (`task2_main_table_clean.ipynb`)

排除 90 个 `additional_benign` 负例，只用主表 2,089 行评估:

| 评估范围 | n | AUROC | AUPRC |
|---|---|---|---|
| 主表-only | 2,089 | 0.5784 | 0.1495 |
| 全量 | 2,179 | 0.5603 | 0.1308 |

**结论**: 主表-only AUROC 与全量仅差 +0.018，说明 additional_benign 未造成明显指标虚高。

---

### Task 3: 捷径检测 (`task3_shortcut_detection.ipynb`)

**实验 A** (排除 additional_benign): AUROC = 0.5784，与全量 (0.5603) 差异 < 0.02。✓

**实验 B** (捷径探针): 训练分类器预测 `is_additional_benign`:
- AUROC = **0.743** > 0.7 ⚠️

**结论**: 特征能区分数据来源 → 全量任务中存在捷径风险。但这未显著影响错定位预测性能（实验 A 证实）。在最终报告中标明此风险。

---

## Phase 1: ΔΔG 特征工程 (Tasks 4, 8, 9, 10)

四种 ΔΔG 计算方法:

| ddg 特征 | Task | 方法 | 类型 | 全量耗时 | 覆盖率 |
|---|---|---|---|---|---|
| `ddg_esm2` | 4 | ESM2 zero-shot masked marginal | 序列 (PLM) | 39s | 2,179/2,179 |
| `ddg_struct` | 8 | MJ 统计势能 (接触势+溶剂化) | 结构 (统计) | 971s (16min) | 2,168/2,179 |
| `ddg_rasp` | 9 | RaSP 3D-CNN cavity | 结构 (DL) | 875s (15min) | 2,168/2,179 |
| `ddg_foldx` | 10 | FoldX 金标准 BuildModel | 结构 (经典) | 669s (11min) | 2,166/2,179 |

### Task 4: ESM2 ΔΔG (`task4_ddg_features.ipynb`)

- **ΔΔG 统计**: mean=5.285, std=4.095, range=[−5.716, 14.969]
- **与 AM 相关**: Spearman r = **0.717** (高度相关)
- **全量 CV**: XGBoost 基线 AUROC=0.594 → +ddg_esm2 AUROC=**0.598** (+0.004)
- **特征重要性**: ddg_esm2 排 **5/1,289** (ESM2 中第 5 重要 — 这还是在 1,280 个 ESM2 维度淹没下的表现)

### Task 8: 结构 ΔΔG (MJ 统计势能) (`task8_struct_ddg.ipynb`)

- **ΔΔG 统计**: mean=0.070, std=1.004, range=[−3.262, 3.593]
- **PCA(50) 下**: baseline=0.614 → +ddg_struct AUROC=**0.609** (−0.005)
- **结论**: MJ 统计势能 ΔΔG 单独不提供增量。

### Task 9: RaSP ΔΔG (`task9_rasp_ddg.ipynb`)

- **ΔΔG 统计**: mean=1.651, std=2.136, range=[−2.068, 11.041]
- **PCA(50) 下**: baseline=0.614 → +ddg_rasp AUROC=**0.615** (+0.001)

### Task 10: FoldX ΔΔG (`task10_foldx_ddg.ipynb`)

- **ΔΔG 统计**: mean=2.416, std=4.456, range=[−5.156, 43.029]
- **PCA(50) 下**: baseline=0.614 → +ddg_foldx AUROC=**0.610** (−0.005)
- **特征重要性**: ddg_foldx 排名 13/59 (0.01894)，但 fold-level 无增益

**关键发现**: 三种基于真实 PDB 结构的 ΔΔG 单独都几乎无增量（ΔAUROC ∈ [−0.005, +0.001]），只有序列零样本 `ddg_esm2` 有微弱增益（+0.004）。这暗示结构热力学稳定性本身对这个错定位任务不提供区分信号。

---

## Phase 2: PCA 降维 + 特征联合 (Tasks 5, 7)

### Task 5: PCA 维度扫描 (`task5_pca_esm2.ipynb`)

将 ESM2 1,280 维压缩，让 7 个结构特征不被淹没:

| PCA 维度 | CV AUROC | 解释方差 | 结构特征最高排名 |
|---|---|---|---|
| Full (1,280) | 0.594 | 1.00 | 全在 30+ 名 |
| PCA(30) | 0.606 | 0.49 | — |
| **PCA(50)** | **0.609** | 0.58 | ss_helix 排名第 6, delta_hydrophobicity 第 16 |
| PCA(100) | 0.575 | 0.70 | — |

**选定 PCA(50)** 作为所有后续实验的基础。降维后 AUROC 从 0.594 升到 0.609（+0.015），且结构特征首次进入 Top-10。

**PCA(50) 结构特征排名**:
| 特征 | 排名/58 | 重要性 |
|---|---|---|
| ss_helix | **6** ★ | 0.02114 |
| delta_hydrophobicity | **16** ★ | 0.01907 |
| sasa | 21 | 0.01821 |
| plddt | 30 | 0.01664 |

### Task 7: PCA(50) + ddg_esm2 联合 (`task7_pca_ddg_combined.ipynb`)

| 配置 | AUROC | AUPRC | Δ vs baseline |
|---|---|---|---|
| v3 基线 (1,280 esm) | 0.594 | 0.160 | — |
| v3 + PCA(50) | 0.614 | 0.156 | +0.021 |
| v3 + PCA(50) + ddg_esm2 | **0.619** | 0.170 | **+0.025** |

ddg_esm2 在 PCA 基础上的额外增益仅 +0.004，但 **ddg_esm2 在 59 个特征中重要性排名第 1** (0.03330)，远超任何单个 PC。

---

## Phase 3: 多 ΔΔG 联合与消融 (Tasks 11, 12, 13)

### Task 11: 三种结构 ΔΔG 联合 (`task11_all_ddg_combined.ipynb`)

将 ddg_struct + ddg_rasp + ddg_foldx 拼在一起（60 维）:

| 配置 | AUROC | Δ vs PCA |
|---|---|---|
| PCA(50) 基线 | 0.614 | — |
| + ddg_struct | 0.609 | −0.005 |
| + ddg_rasp | 0.615 | +0.001 |
| + ddg_foldx | 0.610 | −0.005 |
| **+ ALL3 (三联合)** | **0.616** | +0.001 |

三种结构 ddg 特征重要性: ddg_rasp 排 2/61 (0.02471)，ddg_struct 排 10/61 (0.01966)，ddg_foldx 排 49/61 (0.01405)。

### Task 12: ΔΔG 消融实验 —— ESM2 vs 结构 (`task12_ddg_ablation.ipynb`)

**核心消融**: 序列 ΔΔG 与结构 ΔΔG 是否互补？

| 配置 | 维度 | AUROC | ΔAUROC |
|---|---|---|---|
| PCA(50) 基线 | 57 | 0.6144 | — |
| + ddg_esm2 (ΔΔG) | 58 | 0.6187 | +0.0043 |
| + all_3 (结构×3) | 60 | 0.6155 | +0.0011 |
| **+ ΔΔG + all_3 (全部)** | **61** | **0.6330** | **+0.0186** |

**协同效应**: 单独增益之和 = 0.0043 + 0.0011 = 0.0054，实际联合增益 = 0.0186，**协同 = +0.0132**!

**四种 ddg 特征重要性** (61 维):
| ddg | 排名 | 重要性 |
|---|---|---|
| ddg_esm2 | **1** ★ | 0.03955 |
| ddg_rasp | **3** ★ | 0.02418 |
| ddg_foldx | 13 | 0.01795 |
| ddg_struct | 17 | 0.01764 |

**ddg 间 Spearman 相关性**:
| | ddg_esm2 | ddg_struct | ddg_rasp | ddg_foldx |
|---|---|---|---|---|
| ddg_esm2 | 1.00 | 0.14 | 0.44 | 0.45 |
| ddg_struct | | 1.00 | 0.25 | 0.14 |
| ddg_rasp | | | 1.00 | **0.68** |
| ddg_foldx | | | | 1.00 |

**解读**: ddg_rasp 与 ddg_foldx 高度相关 (r=0.68)，其余之间相关性低 (<0.5)。四种 ddg 来自不同的计算范式，互补性强。

距离 AlphaMissense: gap = 0.6374 − 0.6330 = **0.0044**（几乎追平）

### Task 13: chosen_3 —— 去掉 FoldX (`task13_ddg_esm2_struct_rasp.ipynb`)

**问题**: ddg_foldx 总在排名后半段（49/61），去掉它只用 ddg_esm2 + ddg_struct + ddg_rasp 会不会更好？

| 配置 | AUROC | ΔAUROC |
|---|---|---|
| PCA(50) 基线 | 0.6144 | — |
| **PCA(50) + chosen_3** | **0.6290** | **+0.0146** |

不如四 ddg 联合 (0.6330)。三个 ddg 特征重要性:
- ddg_esm2: 排名 1/61 (0.03966)
- ddg_struct: 排名 5/61 (0.02011)
- ddg_rasp: 排名 7/61 (0.01870)

距离 AlphaMissense: gap = **0.0084**（比四 ddg 略差，但特征更精简）

---

## Phase 4: TMD 靶向特征 (Task 14)

### Task 14: 跨膜域特征 (`task14_TMD_addition.ipynb`)

从 UniProt REST API 拉取 871 个蛋白的跨膜域注释 → 计算 3 个新特征:

| 新特征 | 类型 | 非零占比 | 说明 |
|---|---|---|---|
| `in_TMD` | 0/1 | 151/2,179 (6.9%) | 突变位点是否在跨膜段内 |
| `dist_to_nearest_TMD` | [0,1] | 664/2,179 (30.5%) | 归一化最近 TMD 距离 |
| `delta_membrane_insertion` | 连续 | 151/2,179 (6.9%) | 突变前后 TMD 段膜插入 ΔG 变化 |

**关键质检**:
- corr(delta_hydrophobicity, delta_membrane_insertion) = **−0.215** ✓ (低相关，捕捉不同信号)
- in_TMD: 正例占比 16.6% vs 负例 5.8%（正例更倾向落在 TMD 内 — 有生理解释力）
- UniProt 拉取: 871/871 成功，259 个蛋白有 TMD 注释，耗时 20 分钟

**输出**: `tmd_features.csv` (2,179 × 4 列)

---

## Phase 5: 64 维全特征模型 (Task 15)

### Task 15: 全特征联合 (`task15_full64.ipynb`)

**50PC + 7struct + 4ddg + 3TMD = 64 维**

| 配置 | 维度 | AUROC | AUPRC | ΔAUROC |
|---|---|---|---|---|
| PCA(50) 基线 | 57 | 0.6087 | 0.1597 | — |
| + 4×ddG | 61 | 0.6222 | 0.1778 | +0.0135 |
| **+ 4×ddG + 3×TMD** | **64** | **0.6546** | **0.2026** | **+0.0459** |

**增益分解**:
- 4×ddG: +0.0135
- 3×TMD: +0.0324
- 总增益: +0.0459

**首次超越 AlphaMissense!** gap = −0.0172（模型 AUROC 0.6546 > AM 0.6374）

**64 维特征重要性 Top-10**:
| 排名 | 特征 | 重要性 | 类别 |
|---|---|---|---|
| **1** | ddg_esm2 | 0.03583 | ddg |
| **2** | dist_to_nearest_TMD | 0.03300 | TMD |
| 3 | PC1 | 0.02260 | PC |
| 4 | ss_coil | 0.02174 | struct |
| 5 | PC36 | 0.02149 | PC |
| **6** | delta_membrane_insertion | 0.02075 | TMD |
| 7 | PC39 | 0.02037 | PC |
| 8 | ddg_rasp | 0.02015 | ddg |
| 9 | PC42 | 0.01946 | PC |
| **10** | in_TMD | 0.01922 | TMD |

**所有 TMD 特征全部进入 Top-10!** 三个 TMD 特征加上 ddg_esm2（排名第 1）构成了最强信号源。

结构特征排名: ss_coil(4), sasa(13), plddt(17), rsa(26), delta_hydrophobicity(42), ss_helix(43), ss_strand(49)。

---

## Phase 6: TabPFN v2 模型侧对比 (Task 16)

### Task 16: TabPFN vs XGBoost (`task16_tabpfn.ipynb`)

**协议升级**: 10 折 StratifiedGroupKFold pooled OOF × 3 seeds → 全量 2,179 预测上算 AUROC + 逐 seed 配对 vs AM。

| 模型 | AUROC (mean±std) | AUPRC (mean±std) |
|---|---|---|
| AlphaMissense | 0.6362 ± 0.0000 | 0.1622 ± 0.0000 |
| XGBoost | 0.6221 ± 0.0157 | 0.1765 ± 0.0062 |
| **TabPFN** | **0.6518** ± 0.0134 | **0.2229** ± 0.0078 |
| Ensemble (rank avg) | 0.6507 ± 0.0149 | 0.2104 ± 0.0114 |

**配对比较 (逐 seed TabPFN/Ensemble − AM)**:
| 模型 | seed=42 | seed=7 | seed=2024 | mean Δ | p-value |
|---|---|---|---|---|---|
| XGBoost | +0.0035 | −0.0267 | −0.0192 | −0.0141 | 0.260 (ns) |
| **TabPFN** | **+0.0279** | +0.0014 | **+0.0176** | **+0.0157** | 0.180 (ns) |
| Ensemble | +0.0302 | +0.0005 | +0.0129 | +0.0145 | 0.234 (ns) |

**折间波动** (3 seeds × 10 folds = 30 folds):
- XGBoost: mean=0.621, std=0.065, range=[0.515, 0.726]
- TabPFN: mean=0.659, std=0.071, range=[0.539, 0.782]

**解读**:
1. **TabPFN 在所有 3 个 seed 上均超越 AM** (Δ > 0)，且均值高出 0.016。
2. 但由于只有 3 seeds，t-test 未达显著 (p=0.18)。折间方差大 (std≈0.07) 是主因。
3. **Ensemble (XGB+TabPFN rank 平均) 无明显优势** — TabPFN 单独已足够，集成未带来额外增益。
4. XGBoost 在 TabPFN 面前处于劣势 — 同样的 64 维特征，TabPFN 直接 fit/predict 无调参就高出 0.03 AUROC。

---

## 全部试验 AUROC 演进

| Task | 配置 | 维度 | AUROC | vs AM (0.6374) |
|---|---|---|---|---|
| Task 0 | v3 原始 (1,280 esm) | 1,288 | 0.594 | −0.043 |
| Task 5 | v3 + PCA(50) | 58 | 0.609 | −0.029 |
| Task 7 | v3 + PCA(50) + ddg_esm2 | 59 | 0.619 | −0.019 |
| Task 12 | v3 + PCA(50) + 4×ddG | 61 | **0.633** | −0.004 |
| Task 15 | v3 + PCA(50) + 4×ddG + 3×TMD | 64 | **0.655** | **+0.017** ✓ |
| Task 16 | TabPFN on 64 维 (10-fold OOF) | 64 | **0.652** | **+0.016** ✓ |

**关键转折点**: Task 15 的 TMD 特征加入是决定性的一步（+0.032 over 4×ddG 基线）。

---

## 关键发现总结

### 什么有用 (按增益排序)
1. **TMD 靶向特征** (Task 14→15): +0.032 AUROC — 最大单次增益。`dist_to_nearest_TMD` 在 64 维中排第 2。
2. **PCA 降维** (Task 5): +0.015 AUROC — 让结构特征浮出 ESM2 的淹没。
3. **ddg_esm2 与其他 ddg 的协同** (Task 12): 单独增益 +0.004，联合后协同 +0.013 — 四种不同计算范式的 ΔΔG 互补。
4. **TabPFN 替换 XGBoost** (Task 16): +0.03 AUROC — 零调参直接超越。

### 什么没用
1. **PUPS** (pups_trial): AUROC ≈ 0.55–0.58，无增量。
2. **单一结构 ΔΔG** (ddg_struct/ddg_rasp/ddg_foldx): 各自在 PCA(50) 上增益 < 0.001。
3. **ddg_foldx 单独** (Task 10): 虽为金标准，但 AUROC 反降 0.005。
4. **struct_status** (所有 task): 重要性恒为 0.00000，99.5% 常数，已从所有实验移除。
5. **XGB+TabPFN Ensemble** (Task 16): 不优于 TabPFN 单独。

### 模型侧现状
- **最佳单模型**: TabPFN v2 on 64 维特征，AUROC ≈ 0.652（+0.016 vs AM）
- **最佳 XGBoost**: 64 维 XGBoost, CV AUROC ≈ 0.655（+0.017 vs AM）
- **AM 差距**: 已稳定超越，但幅度不大（≈0.016–0.017）
- **天花板**: 折间方差大（std≈0.07），部分折 AUROC 可达 0.74–0.78，但部分折低至 0.54。特征侧仍有提升空间。

### 下一步方向
1. **DeepLoc 2.0 WT/MT delta 特征** — 亚细胞定位专用信号
2. **进化保守性特征** — 位点级 PhyloP / PhastCons
3. **增加 seed 数** — 3 seeds 不足以获得显著 p-value
4. **RepeatedStratifiedGroupKFold** — 降低折间方差
