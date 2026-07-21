"""从 cell2024 完整数据集中按5分类均衡抽样25个变体, 用于快速门控测试."""

import pandas as pd

SRC = "/mnt/volume6/czj/labLGN/LabLZ/data_preparation/cell2024_final_with_labels.csv"
OUT = "/mnt/volume6/czj/labLGN/LabLZ/pups_trial/gating_variants.csv"

df = pd.read_csv(SRC)
df = df.rename(columns={"sequence": "wt_seq", "mutant_sequence": "mt_seq"})

# 清洗：去空序列/标签、去 WT==MT 的行（无真正突变）
df = df.dropna(subset=["wt_seq", "mt_seq", "label_5class"]).copy()
df["wt_seq"] = df["wt_seq"].str.strip().str.upper()
df["mt_seq"] = df["mt_seq"].str.strip().str.upper()
df = df[df["wt_seq"] != df["mt_seq"]]

# 二值标签：C1=不重定位(0), C2-C5=重定位(1)
df["reloc"] = (df["label_5class"] != "C1_no_reloc").astype(int)

# 按类别均衡抽样（C1 放宽到9例，其余每类4例）
CAPS = {"C1_no_reloc": 9, "C2_aggregation": 4, "C3_secretory": 4,
        "C4_nuclear": 4, "C5_cytoplasmic": 4}
parts = []
for cls, cap in CAPS.items():
    sub = df[df["label_5class"] == cls]
    if len(sub):
        parts.append(sub.sample(min(cap, len(sub)), random_state=0))

out = pd.concat(parts).sample(frac=1, random_state=0).reset_index(drop=True)
out = out[["Gene", "wt_seq", "mt_seq", "reloc", "label_5class"]]
out.to_csv(OUT, index=False)

print(f"写出: {OUT}  共 {len(out)} 个变体")
print(out["label_5class"].value_counts().to_string())
print("重定位/不重定位:", out["reloc"].value_counts().to_dict())
