# 该文件为测试用
import numpy as np
from exceptions.myException import NoPositiveError
a = np.array([1,2,3,4,5])


def get_three(i):
    if i == 3:
        raise NoPositiveError



for i in a:
    try:
        get_three(i)
    except NoPositiveError:
        continue
    print(i)


