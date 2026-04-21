import csv

input_file = 'offground.csv'
output_file = 'offground.csv'

# 读取 CSV 文件
with open(input_file, 'r', newline='') as f_in:
    reader = list(csv.reader(f_in))
    header = reader[0]
    data_rows = reader[1:]

# 找出末尾连续为 0 的列索引
# 从最后一列开始往前找，只要最后一行为 0 就删
num_cols = len(header)
cols_to_keep = num_cols  # 默认保留列数
for i in range(num_cols - 1, -1, -1):
    # 检查所有数据行的该列是否都是 0
    if all(float(row[i]) == 0 for row in data_rows):
        cols_to_keep -= 1
    else:
        break  # 遇到非 0 列就停止

# 截取要保留的列
header_new = header[:cols_to_keep]
data_rows_new = [row[:cols_to_keep] for row in data_rows]

# 写入新 CSV
with open(output_file, 'w', newline='') as f_out:
    writer = csv.writer(f_out)
    writer.writerow(header_new)
    writer.writerows(data_rows_new)

print(f"处理完成，结果已保存到 {output_file}")