import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pups_inference import load_model, predict, CLASSES

CSV = "/mnt/volume6/czj/labLGN/LabLZ/pups_trial/gating_variants.csv"
LM  = "/mnt/volume6/czj/labLGN/LabLZ/pups_trial/real_landmark.npy"


def find_mutation_pos(wt_seq, mt_seq):
    """Return the 1-indexed position where wt_seq and mt_seq first differ."""
    for i, (a, b) in enumerate(zip(wt_seq, mt_seq)):
        if a != b:
            return i + 1
    return None


df = pd.read_csv(CSV)
lm = np.load(LM)
model = load_model()

recs = []
for _, r in df.iterrows():
    pos = find_mutation_pos(r.wt_seq, r.mt_seq)
    _, pwt = predict(model, r.wt_seq, lm, pos=pos)
    _, pmt = predict(model, r.mt_seq, lm, pos=pos)
    d = pmt - pwt
    recs.append({"gene": r.Gene, "cls": r.label_5class, "reloc": int(r.reloc),
                 "l1": float(np.abs(d).sum()),
                 "top": CLASSES[int(np.abs(d).argmax())],
                 "top_delta": float(d[np.abs(d).argmax()])})
res = pd.DataFrame(recs)

print(res.sort_values("l1", ascending=False).to_string(index=False))
print("\n=== 各类平均 |Δ| ===")
print(res.groupby("cls").l1.mean().sort_values(ascending=False))
print("\n② 重定位组 vs 不重定位组 平均|Δ|:",
      round(res[res.reloc == 1].l1.mean(), 4), "vs",
      round(res[res.reloc == 0].l1.mean(), 4))
if res.reloc.nunique() > 1:
    print("   AUROC(|Δ| -> reloc):", round(roc_auc_score(res.reloc, res.l1), 3))
res.to_csv("/mnt/volume6/czj/labLGN/LabLZ/pups_trial/gating_results.csv", index=False)
print("\n结果已存 gating_results.csv")
