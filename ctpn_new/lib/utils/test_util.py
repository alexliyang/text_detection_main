import numpy as np
import cv2
from ..utils.blob import im_list_to_blob

class test_util(object):
    def __init__(self,cfg):
        self._cfg = cfg;
    def _get_image_blob(self,im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self._cfg.TRAIN.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self._cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self._cfg.TEST.MAX_SIZE:
                im_scale = float(self._cfg.TEST.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)


    def _get_blobs(self,im, rois):
        blobs = {'data' : None, 'rois' : None}
        blobs['data'], im_scale_factors = self._get_image_blob(im)
        return blobs, im_scale_factors


    def test_ctpn(self,sess, net, im, boxes=None):
        blobs, im_scales = self._get_blobs(im, boxes)
        if self._cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        # forward pass
        if self._cfg.TEST.HAS_RPN:
            feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

        rois = sess.run([net.get_output('rois')[0]],feed_dict=feed_dict)
        rois=rois[0]

        scores = rois[:, 0]
        if self._cfg.TEST.HAS_RPN:
            assert len(im_scales) == 1, "Only single-image batch implemented"
            boxes = rois[:, 1:5] / im_scales[0]
        return scores,boxes
