import numpy as np

# 生成 x 值
x = np.arange(-1e2, 1e2, 0.01)
x2 = x**2

# 根据公式计算 y 值
y = (1/2) * 9.8 * x**2
y2 = (1/2) * 9.8 * x2
yt = 2 * x

# 创建结构化数组
data = np.column_stack((x, y))
data2 = np.column_stack((x2, y2))
data_t = np.column_stack((x, yt))

# 将 x 和 y 写入文件
np.savetxt("G-small_data.txt", data, fmt='%.6f', delimiter=' ')
