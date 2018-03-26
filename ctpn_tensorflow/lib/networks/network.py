# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from ..fast_rcnn.config import cfg
from ..rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from ..rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py


DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # 有name则返回name， 没name则创建一个name
        """
        这里本来的代码是：
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        而没有name = kwargs['name']
        """
        # name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        name = kwargs['name']
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:   # 这里确实会执行
            layer_input = list(self.inputs)

        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # 把该层的输出结果添加进self.layers
        self.layers[name] = layer_output
        # 喂入进去
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated


class Network(object):
    def __init__(self,  trainable=True): # 这里我把行参inputs去掉列，可以正常运行
        self.inputs = []

        # self.layers是一个字典= dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes,\
        #                   'gt_ishard': self.gt_ishard, 'dontcare_areas': self.dontcare_areas})
        self.layers = dict()           # 这里，本来是dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):

        # data_dict是一个字典， 键为“conv5_1”, "conv3_2"等等
        # 而该字典的值又是字典，键为”biases"和"weights"
        data_dict = np.load(data_path, encoding='latin1').item()
        for key in data_dict.keys():

            # key 是“conv5_1”, "conv3_2"等等
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print("assign pretrain model "+subkey+ " to "+key)
                    except ValueError:
                        print("ignore "+key)
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        # 喂入的数据，不能为空
        assert len(args) != 0
        # 喂入数据之前，先将inputs清空
        self.inputs = []
        for layer in args:
            # 若layer 是字符串类型，则是一个键，应该把值装进inputs
            # 若不是键，是上一层的输出值，则直接把值装进inputs
            if isinstance(layer, str):
                try:
                    layer = self.layers[layer]
                    print(layer)
                except KeyError:
                    print(list(self.layers.keys()))
                    raise KeyError('Unknown layer name fed: %s' % layer)

            self.inputs.append(layer)

        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print(list(self.layers.keys()))
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        # 返回prefix在self.layers.items()中出现的次数，再加一
        i = sum(t.startswith(prefix) for t, _ in list(self.layers.items()))+1
        return '%s_%d' % (prefix, i)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')


    @layer
    def Bilstm(self, input, d_i, d_h, d_o, name, trainable=True):
        # d_i是输入层维度512，d_h是隐层维度128，d_o是输出层维度512
        # 这里的input是由3×3的卷积核在512张通道图片中提取的特征， 是一个1×H×W×512的矩阵
        img = input
        with tf.variable_scope(name) as scope:
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
            weights = self.make_var('weights', [2*d_h, d_o], init_weights, trainable,
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            # 全链接
            outputs = tf.matmul(lstm_out, weights) + biases
            outputs = tf.reshape(outputs, [N, H, W, d_o])
            return outputs

    @layer
    def lstm(self, input, d_i,d_h,d_o, name, trainable=True):
        img = input
        with tf.variable_scope(name) as scope:
            shape = tf.shape(img)
            N,H,W,C = shape[0], shape[1],shape[2], shape[3]
            img = tf.reshape(img,[N*H,W,C])
            img.set_shape([None,None,d_i])

            lstm_cell = tf.contrib.rnn.LSTMCell(d_h, state_is_tuple=True)
            initial_state = lstm_cell.zero_state(N*H, dtype=tf.float32)

            lstm_out, last_state = tf.nn.dynamic_rnn(lstm_cell, img,
                                               initial_state=initial_state,dtype=tf.float32)

            lstm_out = tf.reshape(lstm_out,[N*H*W,d_h])


            init_weights = tf.truncated_normal_initializer(stddev=0.1)
            init_biases = tf.constant_initializer(0.0)
            weights = self.make_var('weights', [d_h, d_o], init_weights, trainable, \
                              regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)
            outputs = tf.matmul(lstm_out, weights) + biases

            outputs = tf.reshape(outputs, [N,H,W,d_o])
            return outputs

    @layer
    def lstm_fc(self, input, d_i, d_o, name, trainable=True):
        with tf.variable_scope(name) as scope:
            shape = tf.shape(input)
            N, H, W, C = shape[0], shape[1], shape[2], shape[3]
            # input的每一行代表一个像素， 第一次C=512
            input = tf.reshape(input, [N*H*W, C])

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [d_i, d_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [d_o], init_biases, trainable)

            _O = tf.matmul(input, kernel) + biases
            return tf.reshape(_O, [N, H, W, int(d_o)])

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,
             relu=True, padding=DEFAULT_PADDING, trainable=True):
        # input是上一层的数据，k_h, k_w为卷集核的高和宽， c_o为输出通道数，s_h, s_w为stride的高和宽
        #
        #  padding 只能为“SAME”或者“VALID”
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]  # 返回输入通道数
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            # 定义一个均值为零， 标准差为0.01的初始化器
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # 定义常量0.0 的初始化器
            init_biases = tf.constant_initializer(0.0)
            # 相当于 tf.Variable() 这里cfg.TRAIN.WEIGHT_DECAY = 0.0005
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable,
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))

            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias, name=scope.name)
                return tf.nn.bias_add(conv, biases, name=scope.name)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv, name=scope.name)
                return conv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
            # input[0] shape is (1, H, W, Ax2)
            # rpn_rois <- (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        with tf.variable_scope(name) as scope:
            blob,bbox_delta = tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales],\
                                     [tf.float32,tf.float32])

            rpn_rois = tf.convert_to_tensor(tf.reshape(blob,[-1, 5]), name = 'rpn_rois') # shape is (1 x H x W x A, 2)
            rpn_targets = tf.convert_to_tensor(bbox_delta, name = 'rpn_targets') # shape is (1 x H x W x A, 4)
            self.layers['rpn_rois'] = rpn_rois
            self.layers['rpn_targets'] = rpn_targets

            return rpn_rois, rpn_targets


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

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            #
            # transpose: (1, AxH, W, 2) -> (1, 2, AxH, W)
            # reshape: (1, 2xA, H, W)
            # transpose: -> (1, H, W, 2xA)
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                            [   input_shape[0],
                                                int(d),
                                                tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),
                                                input_shape[2]
                                            ]),
                                 [0,2,3,1],name=name)
        else:
             return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),
                                        [   input_shape[0],
                                            int(d),
                                            tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),
                                            input_shape[2]
                                        ]),
                                 [0,2,3,1],name=name)

    @layer
    def spatial_reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
        return tf.reshape(input, [input_shape[0], input_shape[1], -1, int(d)])


    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input,name=name)

    @layer
    def spatial_softmax(self, input, name):
        input_shape = tf.shape(input)
        # d = input.get_shape()[-1]
        return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                          [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1])

    @layer
    def batch_normalization(self,input,name,relu=True,is_training=False):
        """contribution by miraclebiu"""
        if relu:
            temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                # tf.nn.l2_loss(t)的返回值是output = sum(t ** 2) / 2
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign +\
                   (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def build_loss(self, ohem=False):
        # 这一步输出的只是分数，没有softmax， 形状为(HxWxA, 2)
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])

        # self.get_output('rpn-data')[0]是形如(1, FM的高，FM的宽，10)的labels
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])  # shape (HxWxA)

        # 取出标签为1 的label
        fg_keep = tf.equal(rpn_label, 1)

        # 保留的标签
        rpn_keep = tf.where(tf.not_equal(rpn_label, -1))

        # 取出保留的标签所在行的分数
        rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)  # shape (N, 2)

        # 取出保留的标签所在的标签
        rpn_label = tf.gather(rpn_label, rpn_keep)

        # 交叉熵损失
        rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)

        rpn_bbox_pred = self.get_output('rpn_bbox_pred')  # shape [1, H, W, 40]

        # rpn_bbox_targets 是(1, FM的高，FM的宽，40) 最后一个维度中，每四个表示一个anchor的回归 x,y,w,h
        rpn_bbox_targets = self.get_output('rpn-data')[1]
        rpn_bbox_inside_weights = self.get_output('rpn-data')[2]
        rpn_bbox_outside_weights = self.get_output('rpn-data')[3]

        rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)

        rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)
        rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
        rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

        # 有内权重，是因为只需计算y值和高度的回归;有外权重，是因为只需计算正例的box回归
        rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * self.smooth_l1_dist(
            rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])

        rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
        rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

        model_loss = rpn_cross_entropy + rpn_loss_box

        # 把正则化项取出来，以列表形式返回
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n(regularization_losses) + model_loss

        # 返回总损失（分类的交叉熵+盒子回归+正则化项）， 模型损失（分类的交叉熵+盒子回归）， 分类交叉熵， 盒子回归损失
        return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
