import numpy as np
import matplotlib.pyplot as plt

# 加载numpy文件
data = np.load('final_1.npy')  # 把 'your_file.npy' 替换为你的文件名


# 检查数据的维度
print(data.shape)  # 应该打印 (3, 1000, 1000)

# 迭代数据的第一个维度
for i in range(data.shape[0]):
    # 获取当前的1000x1000的矩阵
    matrix = data[i, :, :]
    
    # 创建新的图形
    plt.figure(figsize=(6, 6))  # 可以修改大小以适应你的需要
    
    # 使用imshow来显示图像，可以修改cmap以改变颜色映射
    plt.imshow(matrix)
    
    # 添加颜色条
    plt.colorbar()
    
    # 显示图形
    plt.show()

