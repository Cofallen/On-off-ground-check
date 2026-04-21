import numpy as np
import pandas as pd

# 1. 读取数据
df = pd.read_csv('new.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 2. 数据归一化
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1   # 避免除0
X_scaled = (X - mean) / std

# 3. 增加偏置项
X_aug = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

# 4. 超参数
learning_rate = 10
n_iteration = 6000
reg_lambda = 0.1

# 5. 初始化theta
theta = np.random.randn(10) * 0.01

# 6. 训练逻辑回归
for i in range(n_iteration):
    z = X_aug @ theta
    h = 1 / (1 + np.exp(-z))
    
    # 损失函数
    loss = -np.mean(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
    loss += reg_lambda / (2 * X_aug.shape[0]) * np.sum(theta[1:] ** 2)
    
    # 梯度
    gradient = (X_aug.T @ (h - y)) / X_aug.shape[0]
    gradient[1:] += (reg_lambda / X_aug.shape[0]) * theta[1:]
    
    # 更新参数
    theta -= learning_rate * gradient
    
    if i % 1000 == 0:
        print(f"iteration {i}, loss: {loss:.8f}")

# 7. 拆分theta
intercept = theta[0]
coef = theta[1:]

print("train ok")

# 8. 在训练集上计算预测
z_train = X_aug @ theta
y_pred_prob = 1 / (1 + np.exp(-z_train))
y_pred_label = (y_pred_prob >= 0.5).astype(int)

accuracy = np.mean(y_pred_label == y)
print(f"训练集准确率: {accuracy:.4f}")

# 9. 输出C语言格式参数
with open("model_params.c", "w") as f:
    f.write("float w[] = {" + ", ".join([f"{x:.8f}" for x in coef]) + "};\n")
    f.write(f"float b = {intercept:.8f};\n")
    f.write("float mean[] = {" + ", ".join([f"{x:.8f}" for x in mean]) + "};\n")
    f.write("float std[] = {" + ", ".join([f"{x:.8f}" for x in std]) + "};\n")

print("已生成 model_params.c 文件")