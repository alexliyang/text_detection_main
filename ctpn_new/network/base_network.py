import tensorflow as tf
from .anchorlayer.anchor_target_tf import anchor_target_layer_py
DEFAULT_PADDING = "SAME"


# network中方法的专用装饰器
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        name = kwargs['name']

        # 取出输入数据
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # 执行操作，并返回输出数据
        layer_output = op(self, layer_input, *args, **kwargs)
        # 把输出结果加入到layers里面去，保存起来
        self.layers[name] = layer_output
        # 喂入临时缓存
        self.feed(layer_output)
        return self

    return layer_decorated


class base_network(object):
    def __init__(self, cfg):
        self.inputs = []  # 用于存储临时数据
        self.layers = dict()  # 用于存储每一层的数据
        self._cfg = cfg
        self.setup()

    def setup(self):  # 该类不能实例化，必须为子类继承
        raise NotImplementedError('Must be subclassed.')

    def feed(self, *args):
        assert len(args) != 0, "the data to feed cannot be empty!"
        self.inputs = []  # 每次喂入数据前，先将缓存清空
        for _layer in args:
            if isinstance(_layer, str):  # 从子类喂入
                data = self.layers[_layer]
                self.inputs.append(data)
            else:                        # 从装饰器中喂入
                self.inputs.append(_layer)
        return self

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,
             relu=True, padding=DEFAULT_PADDING, trainable=True):
        # input是上一层的数据，k_h, k_w为卷集核的高和宽， c_o为输出通道数，s_h, s_w为stride的高和宽
        c_i = input.get_shape()[-1]  # 返回输入通道数

        with tf.variable_scope(name) as scope:
            # 定义一个均值为零， 标准差为0.01的初始化器
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # 定义常量0.0 的初始化器
            init_biases = tf.constant_initializer(0.0)
            # 相当于 tf.Variable() 这里cfg.TRAIN.WEIGHT_DECAY = 0.0005

            kernel = tf.get_variable(name='weights', shape=[k_h, k_w, c_i, c_o], initializer=init_weights,
                                     trainable=trainable, regularizer=self.l2_regularizer(self._cfg.TRAIN.WEIGHT_DECAY))

            if biased:
                biases = tf.get_variable(name='biases', shape=[c_o], initializer=init_biases, trainable=trainable)
                conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = tf.nn.conv2d(input, kernel, [1, s_h, s_w, 1], padding=padding)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @staticmethod
    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay, dtype=tensor.dtype.base_dtyp, name='weight_decay')
                # tf.nn.l2_loss(t)的返回值是output = sum(t ** 2) / 2
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer


    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        # d_i是输入层维度512，d_h是隐层维度128，d_o是输出层维度512
        # 这里的input是由3×3的卷积核在512张通道图片中提取的特征， 是一个1×H×W×512的矩阵
        img = input
        with tf.variable_scope(name):
            shape = tf.shape(img)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            img = tf.reshape(img, [N * H, W, C])
            img.set_shape([None, None, d_i])   # 第一次 d_i = 512

            lstm_fw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            # 第一次d_h是128, 隐层的维度
            lstm_bw_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)

            # lstm_out是输出， last_state是隐层状态.lstm_out是一个元组，包含前向输出和后向输出(output_fw, output_bw)
            # output_fw和output_bw都是一个形状为[?, ?, 128]的张量
            lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, img, dtype=tf.float32)

            # 将output_fw和output_bw合并成一个[?, ?, 256]的张量
            lstm_out = tf.concat(lstm_out, axis=-1)

            # 每一行对应一个像素，即一个特征每个特征由256维向量表示，lstm_out是一个H×256的输出
            lstm_out = tf.reshape(lstm_out, [N * H * W, 2*d_h])

            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            # 初始化权重，权重是需要正则化的
            weights = tf.get_variable(name='weights', shape=[2*d_h, d_o], initializer=init_weights,
                                      trainable=trainable, regularizer=self.l2_regularizer(self._cfg.TRAIN.WEIGHT_DECAY))
            # 偏执不需要正则化
            biases = tf.get_variable(name='biases', shape=[d_o], initializer=init_biases, trainable=trainable)

            # 全链接
            outputs = tf.nn.bias_add(tf.matmul(lstm_out, weights), biases)
            return tf.reshape(outputs, [N, H, W, d_o])

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name):
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            # input的每一行代表一个像素， 第一次C=512
            input = tf.reshape(input, [N*H*W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)

            weights = tf.get_variable(name='weights', shape=[d_i, d_o], initializer=init_weights,
                                      trainable=trainable, regularizer=self.l2_regularizer(
                    self._cfg.TRAIN.WEIGHT_DECAY))
            # 偏置不需要正则化
            biases = tf.get_variable(name='biases', shape=[d_o], initializer=init_biases, trainable=trainable)

            out = tf.matmul(input, weights) + biases
            return tf.reshape(out, [N, H, W, int(d_o)])






    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        # input里面装着'rpn_cls_score', 'gt_boxes', 'im_info'
        # _feat_stride = [16,], anchor_scales = [16]
        # input的最后一个维度必须是3 ，即'rpn_cls_score', 'gt_boxes', 'im_info'
        assert input.get_shape()[-1] == 3
        if isinstance(input[0], tuple):  # 这里if语句在前期训练阶段没看到成立
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'im_info'

            """
            rpn_labels是(1, FM的高，FM的宽，10),其中约150个值为0,表示正例; 150个值为1表示负例;其他的为-1,不用于训练

            rpn_bbox_targets 是(1, FM的高，FM的宽，40), 最后一个维度中，每四个表示一个anchor的回归 x,y,w,h

            """
            rpn_labels, rpn_bbox_targets = tf.py_func(anchor_target_layer_py,
                                                      [input[0], input[1], input[2],
                                                       _feat_stride, anchor_scales],
                                                      [tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')


            # 这里暂时只需要返回标签和anchor回归目标就可以了，不需要添加内部外部权重，后续会增加side refinement
            return rpn_labels, rpn_bbox_targets




