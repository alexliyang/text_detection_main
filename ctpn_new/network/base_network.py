import tensorflow as tf

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
                                     trainable=trainable, regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))

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
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        # input里面装着'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'
        # _feat_stride = [16,], anchor_scales = [16]
        if isinstance(input[0], tuple):  # 这里if语句在前期训练阶段没看到成立
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            # 'rpn_cls_score', 'gt_boxes', 'gt_ishard', 'dontcare_areas', 'im_info'

            """
            rpn_labels是(1, FM的高，FM的宽，10),其中约150个值为0,表示正例; 150个值为1表示负例;其他的为-1,不用于训练

            rpn_bbox_targets 是(1, FM的高，FM的宽，40), 最后一个维度中，每四个表示一个anchor的回归 x,y,w,h

            rpn_bbox_inside_weights (1, FM的高，FM的宽，40)， 最后一个维度中，每四个表示一个anchor的内部权重，
            标签为0的是0,0,0,0; 标签为1的是0,1,0,1, 其余的全0

            rpn_bbox_outside_weights(1, FM的高，FM的宽，40)， 最后一个维度中，每四个表示一个anchor的外部权重，
            标签为0的是0,0,0,0; 标签为1的是1,1,1,1, 其余的全0

            且不知内部权重和外部权重是干啥的
            """
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0], input[1], input[2], input[3], input[4], _feat_stride, anchor_scales],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


