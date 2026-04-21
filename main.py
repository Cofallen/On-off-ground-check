import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from micromlgen import port

# 加载数据
df = pd.read_csv('new.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 可选：标准化（随机森林不需要，但保持与之前一致）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
mean = scaler.mean_
std = scaler.scale_
std[std == 0] = 1.0

# 训练随机森林（调整参数以获得更好性能）
rf = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=8,           # 限制深度，防止过拟合
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_scaled, y)

# 训练集损失
y_pred = rf.predict_proba(X_scaled)[:, 1]
loss = log_loss(y, y_pred)
print(f"随机森林训练集 Log-Loss: {loss:.6f}")
print(f"准确率: {rf.score(X_scaled, y):.4f}")

# # 导出为 C 代码
# c_code = port(rf, classmap={0: "off", 1: "on"})
# print("\n/* ----- 随机森林 C 代码 ----- */")
# print(c_code)

# 同时输出标准化参数（如果你需要在 C 中先标准化输入）
print("\n/* ----- 标准化参数（用于预处理输入）----- */")
print("double mean[] = {", ", ".join([f"{x:.8f}" for x in mean]), "};")
print("double std[] = {", ", ".join([f"{x:.8f}" for x in std]), "};")
print("/* 使用前请将每个特征标准化: (x[i] - mean[i]) / std[i] */")