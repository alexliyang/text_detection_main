'''
InputLayer是整个网络的输入层，在每iter中，需要获取下一个batch的数据，其核心函数是forward
'''

class InputLayer(object):
	def __init__(self):
		pass

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        pass


    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        pass
        

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        pass
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        pass
