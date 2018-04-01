import numpy.random as npr
import numpy as np
import cv2
import os


def get_minibatch(roidb, cfg):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images

    # Get the input image blob, formatted for caffe
    im_blob, im_scales = _get_image_blob(roidb, random_scale_inds, cfg)

    single_blob = {'data': im_blob,
                   # ”gt_boxes"须是一个N行4列的矩阵，每一行代表一个GT
                   'gt_boxes': np.array(roidb[0]['boxes']),
                   # im_info须是一个包含三个元素的向量，分别代表图片的高，宽，缩放比
                   'im_info': np.array([im_blob.shape[1], im_blob.shape[2], roidb.image_scale]),
                   'im_name': os.path.basename(roidb[0]['image_name'])
                   }
    return single_blob


def _get_image_blob(roidb, scale_inds, cfg):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)

    assert num_images == 1, "image batch size must be one"

    im = cv2.imread(roidb[0]['image_path'])

    target_size = int(cfg.TRAIN.SCALES[0])
    im, im_scale = prep_im_for_blob(im, np.array(cfg.TRAIN.PIXEL_MEANS), target_size,
                                    cfg.TRAIN.MAX_SIZE, cfg)
    # Create a blob to hold the input images
    blob = np.zeros((num_images, im.shape[0], im.shape[1], 3),
                    dtype=np.float32)

    blob[0, :, :, :] = im
    # blob是一张图片，形状为[1,高，宽，3], im_scale是其缩放比,float
    return blob, im_scale


def prep_im_for_blob(im, pixel_means, target_size, max_size, cfg):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)

    # TODO 这里是用预处理以后的图片计算缩放比，可能会有问题
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    return im, im_scale
