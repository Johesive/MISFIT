import numpy as np
from pups_inference import load_model, predict, CLASSES

WT = "MKT...(你的WT序列)..."
MT = "MKT...(同蛋白单点突变)..."

model = load_model()
print("模型已加载, device:", next(model.parameters()).device)

lm = np.random.rand(3, 128, 128).astype("float32")   # 先用随机 landmark 只验证链路/权重/形状
img_wt, p_wt = predict(model, WT, lm)
img_mt, p_mt = predict(model, MT, lm)

print("预测图形状:", img_wt.shape, " 期望 (128,128)")
print("多标签形状:", p_wt.shape, " 期望 (29,)")
print("WT/MT 是否不同:", not np.allclose(p_wt, p_mt))   # 序列通路通了就应为 True
d = np.abs(p_mt - p_wt)
print("变化最大的区室:", CLASSES[d.argmax()], f"{d.max():.3f}")
