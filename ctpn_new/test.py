#coding:utf-8
#在这个文件里调用test_net.py中的test_net
import sys
import os
sys.path.append(os.getcwd())
import pprint
from network.test_network import get_test_network
from lib.load_config import load_config
from ctpn.test_net import TestClass


if __name__ == "__main__":
    #加载配置文件
    cfg = load_config();
    #pprint.pprint(cfg)

    #获取测试网络
    network = get_test_network(cfg)

    #获取测试类实例
    testclass = TestClass(cfg,network)
    #开始测试
    testclass.test_net();

