"""
InputLayer是整个网络的输入层，在每iter中，需要获取下一个batch的数据，其核心函数是forward
"""
import numpy as np
from .minibatch import get_minibatch


class InputLayer(object):
    def __init__(self, roidb, num_classes=2, config=None):
        if config == None:
            raise RuntimeError('Input layer lack config')
        self._cfg = config
        self._roidb = roidb.roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        cfg = self._cfg
        if self._cur + cfg.TRAIN.IMS_BATCH_SIZE >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_BATCH_SIZE]
        self._cur += cfg.TRAIN.IMS_BATCH_SIZE
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes, cfg=self._cfg)

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        return self._get_next_minibatch()


def get_data_layer(roidb, config):
    return InputLayer(roidb, config=config)
