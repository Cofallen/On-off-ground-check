import pandas as pd

# 读取原csv文件
df_on = pd.read_csv('onground.csv')  # 替换为你的文件名
# 在每行最后加一列，值为1
df_on['new_col'] = 1
# 保存为新文件
df_on.to_csv('onground.csv', index=False)  # 保存为新文件

# 读取原csv文件
df_off = pd.read_csv('offground.csv')  # 替换为你的文件名
# 在每行最后加一列，值为1
df_off['new_col'] = 0
# 保存为新文件
df_off.to_csv('offground.csv', index=False)  # 保存为新文件

df = pd.concat([df_on, df_off], axis=0)
df.to_csv('new.csv', index=False)