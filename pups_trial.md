# PUPS 模型试验

> PUPS (Protein Unsupervised Placement Scores) —— 基于 ESM2 嵌入的无监督蛋白质亚细胞定位预测模型

---

## 背景

PUPS 是本项目的对照模型方向之一。它利用 ESM2 的蛋白级 embedding 计算 WT 与突变蛋白之间的"定位位移"向量（Δ|PUPS|），将其作为错定位风险的预测分数。与 XGBoost 路径不同，PUPS 不需要训练 —— 它直接输出一个分数，因此可以作为独立基线。

---

## 实验 1: 全量 PUPS 运行 (v1 标签) (`5_run_pups_full_A.ipynb`)

### 设置

- **标签**: 原始 `Mislocalized` 列（0/1）
- **方法**: 对 2,179 个变体逐一计算 PUPS 位移向量的大小（|Δ|）
- **计算量**: 2,179/2,179 全部完成

### 结果

| 指标 | 值 |
|---|---|
| 评估样本 | 2,179 (正例 235) |
| **AUROC** | **0.553** |

**解读**: PUPS 得分仅略高于随机基线（0.50），单独作为预测器几乎没有区分能力。

---

## 实验 2: 增量测试 (v1 标签) (`5_test_incremental.ipynb`)

### 设置

测试 PUPS 特征是否能给 ESM2 delta embedding 模型带来增量提升。

| 模型 | AUROC |
|---|---|
| PUPS only | 0.526 |
| ESM2-delta only | 0.516 |
| ESM2-delta + PUPS | 0.526 |

**Δ = +0.010** → PUPS 几乎无增量，对 ESM2 delta 特征没有补充信号。

---

## 实验 3: 全量 PUPS 运行 (v2 标签) (`run_pups_full_A_v2.ipynb`)

### 改进

- **v2 标签**: 将 `C1_no_reloc`（同区室重定位，n=13）从正例重标为负例
  - 负例: 1,943 + 13 = 1,956
  - 正例: 222（纯 C2-C5）
- 使用 l1 距离代替 l2 距离

### 结果

| 指标 | 值 |
|---|---|
| **v2 AUROC** | **0.582** |

**解读**: v2 标签修复使 PUPS 从 0.553 提升到 0.582（+0.029），但仍远低于 AlphaMissense（0.636）。

---

## 实验 4: v2 增量测试 (`test_incremental_v2.ipynb`)

### 设置

在 871 个基因的完整样本上测试 PUPS 与 ESM2 delta 的组合效果。

| 模型 | AUROC | AUPRC |
|---|---|---|
| PUPS only | 0.515 | 0.128 |
| ESM2-delta only | 0.533 | 0.121 |
| ESM2-delta + PUPS | 0.543 | 0.126 |

**增量 Δ = +0.010** (AUROC) / +0.005 (AUPRC)

---

## PUPS 试验总结

| 实验 | 标签 | PUPS AUROC | 结论 |
|---|---|---|---|
| Full A | v1 (Mislocalized=235) | 0.553 | 弱于随机+0.05 |
| Incremental | v1 | Δ=+0.010 | 无增量 |
| Full A v2 | v2 (C2-C5=222) | 0.582 | 略有改善 |
| Incremental v2 | v2 | Δ=+0.010 | 仍无显著增量 |

### 核心结论

**PUPS 不能作为有效的错定位预测器。** 即使改用更干净的 v2 标签和 l1 距离，其 AUROC 最高仅 0.582，显著低于 AlphaMissense 的 0.636。更重要的是，PUPS 对 ESM2 delta embedding 几乎不提供增量（Δ ≈ +0.01），说明它捕捉的信号与 ESM2 嵌入差异高度冗余。

**因此，后续工作全面转向了 XGBoost + 结构/进化特征 + ΔΔG 的路径（见 `xgboost_trial.md`）。**
