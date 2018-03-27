import sys
import os
from network.train_network import get_train_network
from ctpn.train_net import train_net
sys.path.append(os.getcwd())
import pprint
from lib.load_config import load_config


if __name__ == '__main__':
    cfg = load_config()
    print('Using config:')
    pprint.pprint(cfg)
    print(cfg.TRAIN.USE_CACHE)


    """
    @params
     use_cache 是否从重新进行data_process过程，一般dataset/for_train文件发生变化需要进行
    """
    roidb = get_training_roidb(cfg)  # 返回roidb roidb就是我们需要的对象实例

    output_dir = ''
    log_dir = ''
    print('Output will be saved to `{:s}`'.format(output_dir))

    print('Logs will be saved to `{:s}`'.format(log_dir))



    # """ 
    # @params 
    # network ctpn_network 实例
    # roidb roi 列表
    # output_dir tensorflow输出的 绝对路径 要拼接本地机器所在的运行目录
    # log_dir 日志输出 绝对路径
    # max_iter 训练轮数
    # pretrain_model 预训练VGG16模型 绝对路径
    # restore bool值 是否从checkpoints断点开始恢复上次中断的训练
    # """

    network = get_train_network(cfg)
    train_net(cfg, network, roidb, output_dir, log_dir, max_iter, pretrain_model, restore=False)




