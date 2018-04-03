import numpy as np
a = np.array(range(24))
a = a.reshape((2,3,4))

b = a[:, 1, :]
c = list(b)
print("b", b)
print("c", c)
c += c
print([list(list(c))])
# print(list(c[0])


