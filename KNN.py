import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

# ----------------------------
# 用户配置
# ----------------------------
input_csv = 'new.csv'         # 数据文件
output_c_file = 'rf_ground_check.c'  # 输出 C 文件
n_trees = 100                 # 随机森林树数量
max_depth = 10                 # 树最大深度（越大生成代码越长）
# ----------------------------

# 1. 读取数据
df = pd.read_csv(input_csv)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
mean = scaler.mean_
std = scaler.scale_
std[std == 0] = 1.0

# 3. 训练随机森林
rf = RandomForestClassifier(
    n_estimators=n_trees,
    max_depth=max_depth,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
rf.fit(X_scaled, y)

# 训练集损失
y_pred = rf.predict_proba(X_scaled)[:, 1]
loss = log_loss(y, y_pred)
print(f"随机森林训练集 Log-Loss: {loss:.6f}")
print(f"准确率: {rf.score(X_scaled, y):.4f}")


# 4. 特征名称映射到 ground_check 样式
feature_names = [
    "leg->LQR.F_0",
    "leg->LQR.T_p",
    "leg->stateSpace.theta",
    "leg->stateSpace.dtheta",
    "leg->stateSpace.dtheta*leg->stateSpace.dtheta",
    "leg->stateSpace.ddtheta",
    "sin(leg->stateSpace.theta)",
    "cos(leg->stateSpace.theta)",
    "imu->accel[0]"
]

# 5. 生成 C 函数
lines = []
lines.append("#include <stdint.h>")
lines.append("#include <math.h>")
lines.append(f"#define N_FEATURES {X.shape[1]}")
lines.append("")
lines.append(f"const float mean[N_FEATURES] = {{ {', '.join(f'{m:.8f}' for m in mean)} }};")
lines.append(f"const float std[N_FEATURES]  = {{ {', '.join(f'{s:.8f}' for s in std)} }};")
lines.append("")
lines.append("uint8_t ground_check(Leg_Typedef *leg, IMU_Data_t *imu) {")
lines.append("    float norm[N_FEATURES];")

# 特征标准化
for i, feat in enumerate(feature_names):
    lines.append(f"    norm[{i}] = ({feat} - mean[{i}]) / std[{i}];")

lines.append("    int votes = 0;")
lines.append("")

# 遍历每棵树
for t_idx, tree in enumerate(rf.estimators_):
    lines.append(f"    // ----- Tree {t_idx} -----")
    tree_ = tree.tree_

    # 递归生成 if/else
    def recurse(node, depth=1):
        indent = "    " * depth
        if tree_.feature[node] == -2:  # 叶子
            value = tree_.value[node][0][0]
            # 随机森林输出是概率或类别计数，这里大于0.5投1，否则0
            lines.append(f"{indent}votes += ({value} >= 0.5) ? 1 : 0;")
            return
        # 内部节点
        feat = tree_.feature[node]
        thresh = tree_.threshold[node]
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        lines.append(f"{indent}if (norm[{feat}] <= {thresh:.6f}f) {{")
        recurse(left, depth+1)
        lines.append(f"{indent}}} else {{")
        recurse(right, depth+1)
        lines.append(f"{indent}}}")

    recurse(0)  # 从根节点开始

lines.append("")
lines.append(f"    return (votes >= {n_trees//2}) ? 1 : 0;")
lines.append("}")

# 保存文件
with open(output_c_file, "w") as f:
    f.write("\n".join(lines))

print(f"嵌入式 C 函数已生成: {output_c_file}")