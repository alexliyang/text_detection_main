import os.path as osp
print(__file__)
print("=============================")
print(osp.dirname(__file__))
print("============================")
a = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
print(a)
