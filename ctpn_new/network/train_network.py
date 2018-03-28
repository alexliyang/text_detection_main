import tensorflow as tf
from .base_network import base_network as bn


class train_network(bn):
    def __init__(self, cfg):
        # super().__init__(cfg)
        # 数据的输入入口
        self._cfg = cfg
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='data')
        # 图像信息，包含宽，高，缩放比例
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3], name='im_info')
        # GT_boxes信息，前四列为缩放后的四个坐标
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 4], name='gt_boxes')
        self.keep_prob = tf.placeholder(tf.float32)
        self.setup()

    def setup(self):
        self.inputs = []
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        anchor_scales = [16]
        _feat_stride = [16, ]

        # padding本来是“VALID”，我把下面的padding全部改为了“SAME”， 以充分检测
        (self.feed('data')   # 把[批数，宽，高，通道]形式的源图像数据喂入inputs
             .conv(3, 3, 64, 1, 1, name='conv1_1')
             .conv(3, 3, 64, 1, 1, name='conv1_2')   # k_h, k_w, c_o, s_h, s_w, name,
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
        # 在conv5_3中做滑动窗， 得到3×3×512的特征向量
        (self.feed('conv5_3')
             .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3'))
        # 将得到3×3×512的特征向量提取10个anchor并做双向LSTM中
        (self.feed('rpn_conv/3x3').bilstm(512, 128, 512, name='lstm_o'))  # 这里的512必须与最后一个卷积层的512匹配
        # Bilstm的输出为[N, H, W, 512]形状

        # 往两个方向走，一个用于给类别打分，一个用于盒子回归
        # 用于盒子回归的，输入是10个anchor，每个anchor有4个坐标，所以输出是[1, H, W, 40]
        (self.feed('lstm_o').lstm_fc(512, 10 * self._cfg.TRAIN.COORDINATE_NUM, name='rpn_bbox_pred'))

        # 用于盒子分类的，输出是[1, H, W, 20]
        (self.feed('lstm_o').lstm_fc(512, 10 * self._cfg.TRAIN.COORDINATE_NUM, name='rpn_cls_score'))

        """
        返回值如下
        rpn_labels是(1, FM的高，FM的宽，10),其中约150个值为0,表示正例; 150个值为1表示负例;其他的为-1,不用于训练
        rpn_bbox_targets 是(1, FM的高，FM的宽，40), 最后一个维度中，每四个表示一个anchor的回归 x,y,w,h

        """
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info')
             .anchor_target_layer(_feat_stride, anchor_scales, name='rpn-data'))

        # shape is (1, H, W, Ax2) -> (1, H, WxA, 2)
        # 给之前得到的score进行softmax，得到0-1之间的得分
        (self.feed('rpn_cls_score')
             .spatial_reshape_layer(2, name='rpn_cls_score_reshape')  # 把最后一个维度变成2,即(1,H,W,Ax2)->(1,H,WxA,2)
             .spatial_softmax(name='rpn_cls_prob'))  # 执行softmax，再转换为(1, H, WxA,2)


def get_train_network(cfg):
    return train_network(cfg)
