import tensorflow as tf
import os
from lib import Timer
from input_layer import get_data_layer


class SolverWrapper(object):
    def __init__(self, cfg, network, roidb, checkpoints_dir, output_dir, log_dir, max_iter, pretrain_model, restore):
        self._cfg = cfg
        self.net = network
        # 所有图片的imdb列表
        self.roidb = roidb  # 所有图片的GT列表，每个元素是一个字典，字典里面包含列所有的box
        self.output_dir = output_dir
        self.pretrained_model = pretrain_model
        self.checkpoints_dir = checkpoints_dir

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100, write_version=tf.train.SaverDef.V2)

    def snapshot(self, sess, iter):

        # if not os.path.exists(self.output_dir):
        #     os.makedirs(self.output_dir)
        #
        # infix = ('_' + self._cfg.TRAIN.SNAPSHOT_INFIX
        #          if self._cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = ('ctpn_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.checkpoints_dir, filename)

        self.saver.save(sess, filename)
        print('Wrote snapshot to: {:s}'.format(filename))

    def train_model(self, sess, max_iters, restore=False):
        # 根据全部的roidb，获得一个data_layer对象
        # data_layer对象是一批一批地传递处理好了的数据
        data_layer = get_data_layer(self.roidb, self._cfg)

        total_loss, model_loss, rpn_cross_entropy, rpn_loss_box = self.net.build_loss()

        # cfg.TRAIN.LEARNING_RATE = 0.00001
        lr = tf.Variable(self._cfg.TRAIN.LEARNING_RATE, trainable=False)
        # TRAIN.SOLVER = 'Momentum'
        if self._cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        elif self._cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(self._cfg.TRAIN.LEARNING_RATE)
        else:
            # lr = tf.Variable(0.0, trainable=False)
            momentum = self._cfg.TRAIN.MOMENTUM  # 0.9
            opt = tf.train.MomentumOptimizer(lr, momentum)

        global_step = tf.Variable(0, trainable=False)
        with_clip = True
        if with_clip:
            tvars = tf.trainable_variables()  # 获取所有的可训练参数
            # 下面这句话会产生UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape.
            # This may consume a large amount of memory
            grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)

            train_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
        else:
            train_op = opt.minimize(total_loss, global_step=global_step)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load vgg16
        if self.pretrained_model is not None and not restore:
            try:
                print(('Loading pretrained model '
                       'weights from {:s}').format(self.pretrained_model))

                # 从预训练模型中导入
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(self.pretrained_model)

        # resuming a trainer
        if restore:  # restore为True表示训练过程中可能死机了， 现在重新启动训练
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('_')[-1])
                sess.run(global_step.assign(restore_iter))
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
        last_snapshot_iter = -1
        timer = Timer()

        for iter in range(restore_iter, max_iters):
            timer.tic()
            # learning rate
            if iter != 0 and iter % self._cfg.TRAIN.STEPSIZE == 0:  # 每30000轮，学习率变为原来的0.1
                sess.run(tf.assign(lr, lr.eval() * self._cfg.TRAIN.GAMMA))
                print("learning rate at step {} is {}".format(iter, lr))

            blobs = data_layer.forward()

            feed_dict = {
                self.net.data: blobs['data'],  # 一个形状为[批数，宽，高，通道数]的源图片，命名为“data”
                self.net.im_info: blobs['im_info'],  # 一个三维向量，包含宽，高，缩放比例
                self.net.keep_prob: 0.5,
                self.net.gt_boxes: blobs['gt_boxes'],  # GT_boxes信息，N×4矩阵，每一行为一个gt_box，分别代表x1,y1,x2,y2
            }

            fetch_list = [total_loss, model_loss, rpn_cross_entropy,
                          rpn_loss_box, train_op]

            total_loss_val, model_loss_val, rpn_loss_cls_val, rpn_loss_box_val, \
            summary_str = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            _diff_time = timer.toc(average=False)

            if iter % self._cfg.TRAIN.DISPLAY == 0:
                print('iter: %d / %d, total loss: %.4f, model loss: %.4f, rpn_loss_cls: %.4f, '
                      'rpn_loss_box: %.4f, lr: %f' % (iter, max_iters, total_loss_val, model_loss_val,
                                                      rpn_loss_cls_val, rpn_loss_box_val, lr.eval()))
                print('speed: {:.3f}s / iter'.format(_diff_time))

            # =====================================================================================================
            # 郭义，到此投笔

            # 每1000次保存一次模型
            if (iter + 1) % self._cfg.TRAIN.SNAPSHOT_ITERS == 0:  # 每一千差一次
                last_snapshot_iter = iter
                self.snapshot(sess, iter)


def train_net(cfg, network, roidb, checkpoints_dir, output_dir, log_dir, max_iter, pretrain_model, restore):
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        '''sw = solver wrapper'''
        sw = SolverWrapper(cfg, network, roidb, checkpoints_dir, output_dir, log_dir, max_iter, pretrain_model, restore)
        print('Solving...')

        sw.train_model(sess=sess, max_iters=max_iter)
        print('done solving')
