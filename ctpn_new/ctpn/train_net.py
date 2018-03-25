import tensorflow as tf

class SolverWrapper(object):
	def __init__(self):
		pass

    def snapshot(self, sess, iter):
    	pass

    def build_image_summary(self):
    	pass

	def train_model(self):
		pass


def train_net():
    """Train a Fast R-CNN network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allocator_type = 'BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        '''sw = solver wrapper'''
        sw = SolverWrapper()
        print('Solving...')

        sw.train_model(sess, max_iters, restore=restore)
        print('done solving')

