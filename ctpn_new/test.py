"""这个文件用于写代码时候的测试"""
import numpy as np
a = np.zeros((10,))
a[[1, 3, 5, 6, 7]] = 1
print(a)

