import numpy as np
import os
a = np.array([0,1,2,3,4,5])
delete_index = []
b = [a[i] for i in range(len(a)) if i not in delete_index]
print(b)