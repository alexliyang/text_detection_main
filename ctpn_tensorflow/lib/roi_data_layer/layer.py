# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import numpy as np

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
# <<<< obsolete
from ..roi_data_layer.minibatch import get_minibatch


class RoIDataLayer(object):
    """Fast R-CNN data layer used for training."""

    def __init__(self, roidb, num_classes):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._num_classes = num_classes
        self._shuffle_roidb_inds()

    # 将训练的roidb索引随机打乱
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    # 获得接下来两个训练样本，即返回下两个样本的索引，并将指标后移两格
    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if cfg.TRAIN.HAS_RPN:         # 这里是True
            # 这里，cfg.TRAIN.IMS_PER_BATCH = 1    不知为啥………………………………………………………………………………………………………………
            if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
                self._shuffle_roidb_inds()  # ？？？？？？？？？？？？？？？？？？？？？？？这里重新洗牌后，依然越界

            # 取一个索引
            db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]  # 这里len(db_inds)=1


            self._cur += cfg.TRAIN.IMS_PER_BATCH
        else:
            # sample images
            db_inds = np.zeros((cfg.TRAIN.IMS_PER_BATCH), dtype=np.int32)
            i = 0
            while (i < cfg.TRAIN.IMS_PER_BATCH):
                ind = self._perm[self._cur]
                num_objs = self._roidb[ind]['boxes'].shape[0]
                if num_objs != 0:
                    db_inds[i] = ind
                    i += 1

                self._cur += 1
                if self._cur >= len(self._roidb):
                    self._shuffle_roidb_inds()

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()  # len(db_inds)=1
        # 根据一个索引，获得指定的一个roidb元素
        minibatch_db = [self._roidb[i] for i in db_inds]
        """根据指定的roidb元素，返回一个blobs
        blobs 是一个字典，其键有 
        data,即[批数，宽，高，通道数]的源图片;
        gt_boxes,所有的GT四个坐标
        gt_ishard
        dontcare_areas
        im_info 图片的宽，高，缩放比例
        im_name 图片的名字
        """
        return get_minibatch(minibatch_db, self._num_classes)
            
    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs
