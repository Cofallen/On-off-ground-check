import numpy as np
import pandas as pd

df = pd.read_csv('new.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

mean = X.mean(axis=0)
std = X.std(axis=0)
std[std == 0] = 1   # 避免/0
X_scaled = (X - mean) / std

X_aug = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

learnig_rate = 0.1
n_iteration = 50000
reg_lambda = 1

theta = np.random.randn(13) * 0.01
# theta = np.zeros(13)  # 初始化为0也可以，随机初始化有时会更快收敛

for i in range(n_iteration):
    z = X_aug @ theta
    h = 1 / (1 + np.exp(-z))
    
    loss = -np.mean(y * np.log(h + 1e-8) + (1 - y) * np.log(1 - h + 1e-8))
    loss += reg_lambda / (2 * X_aug.shape[0]) * np.sum(theta[1:] ** 2)
    
    gradient = (X_aug.T @ (h - y)) / X_aug.shape[0]
    
    gradient[1:] += (reg_lambda / X_aug.shape[0]) * theta[1:]
    
    theta -= learnig_rate * gradient
    if i % 1000 == 0:
        print(f"iteration {i}, loss: {loss:.8f}")
        

intercept = theta[0]
coef = theta[1:]

print("train ok")

# C语言格式输出
print("double w[] = {", ", ".join([f"{x:.8f}" for x in coef]), "};")
print(f"double b = {intercept:.8f};")
print("double mean[] = {", ", ".join([f"{x:.8f}" for x in mean]), "};")
print("double std[] = {", ", ".join([f"{x:.8f}" for x in std]), "};")


with open("model_params.c", "w") as f:
    f.write("float w[] = {" + ", ".join([f"{x:.8f}" for x in coef]) + "};\n")
    f.write(f"float b = {intercept:.8f};\n")
    f.write("float mean[] = {" + ", ".join([f"{x:.8f}" for x in mean]) + "};\n")
    f.write("float std[] = {" + ", ".join([f"{x:.8f}" for x in std]) + "};\n")

print("已生成 model_params.c 文件")