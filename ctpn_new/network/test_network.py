#coding:utf-8
import tensorflow as tf
from .base_network import base_network


class test_network(base_network):
    def _int__(self,cfg):
        self.cfg = cfg
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.inputs = []
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.setup()

    def setup(self):
        anchor_scales = 16
        _feat_stride = [16, ]
        #=========VGG16网络结构,与训练部分相同========
        (self.feed('data')
            .conv(3, 3, 64, 1, 1, name='conv1_1')
            .conv(3, 3, 64, 1, 1, name='conv1_2')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool1')
            .conv(3, 3, 128, 1, 1, name='conv2_1')
            .conv(3, 3, 128, 1, 1, name='conv2_2')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
            .conv(3, 3, 256, 1, 1, name='conv3_1')
            .conv(3, 3, 256, 1, 1, name='conv3_2')
            .conv(3, 3, 256, 1, 1, name='conv3_3')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool3')
            .conv(3, 3, 512, 1, 1, name='conv4_1')
            .conv(3, 3, 512, 1, 1, name='conv4_2')
            .conv(3, 3, 512, 1, 1, name='conv4_3')
            .max_pool(2, 2, 2, 2, padding='SAME', name='pool4')
            .conv(3, 3, 512, 1, 1, name='conv5_1')
            .conv(3, 3, 512, 1, 1, name='conv5_2')
            .conv(3, 3, 512, 1, 1, name='conv5_3'))
        # ========= RPN ============
        #注释详见train_network.py文件
        (self.feed('conv5_3').conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        (self.feed('rpn_conv/3x3').bilstm(512, 128, 512, name='lstm_o'))  # 这里的512必须与最后一个卷积层的512匹配
        (self.feed('lstm_o').lstm_fc(512, 10 * self._cfg.COORDINAE_NUM, name='rpn_bbox_pred'))
        (self.feed('lstm_o').lstm_fc(512, 10 * self._cfg.COORDINAE_NUM, name='rpn_cls_score'))

        #计算分数与回归
        #  shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        (self.feed('rpn_cls_score')
            .spatial_reshape_layer(2, name='rpn_cls_score_reshape')
            .spatial_softmax(name='rpn_cls_prob'))
        # shape is (1, H, WxA, 2) -> (1, H, W, Ax2)
        (self.feed('rpn_cls_prob')
            .spatial_reshape_layer(10 * self._cfg.COORDINAE_NUM, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
            .proposal_layer(_feat_stride, anchor_scales, 'TEST', name='rois'))

def get_test_network(cfg):
    return test_network(cfg)