import numpy.random as npr
import numpy as np
import cv2
import os


def get_minibatch(roidb, num_classes, cfg):
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
                   'gt_boxes': np.array(roidb[0]['boxes']),
                   'im_info': np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]]),
                   'im_name': os.path.basename(roidb[0]['image_name'])
                   }
    return single_blob
    # assert len(im_scales) == 1, "Single batch only"
    # assert len(roidb) == 1, "Single batch only"
    # # gt boxes: (x1, y1, x2, y2, cls)
    # # gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    #
    # gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    # gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
    # blobs['gt_boxes'] = gt_boxes
    # # blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds] \
    # #     if 'gt_ishard' in roidb[0] else np.zeros(gt_inds.size, dtype=int)
    # # # blobs['gt_ishard'] = roidb[0]['gt_ishard'][gt_inds]
    # # blobs['dontcare_areas'] = roidb[0]['dontcare_areas'] * im_scales[0] \
    # #     if 'dontcare_areas' in roidb[0] else np.zeros([0, 4], dtype=float)
    # blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
    #     dtype=np.float32)
    # blobs['im_name'] = os.path.basename(roidb[0]['image'])
    #
    # return blobs



def _get_image_blob(roidb, scale_inds, cfg):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    assert num_images != 1, "image batch size must be one"

    im = cv2.imread(roidb[0]['image_path'])

    target_size = int(cfg.TRAIN.SCALES)
    im, im_scale = prep_im_for_blob(im, np.array(cfg.TRAIN.PIXEL_MEANS), target_size,
                                    cfg.TRAIN.MAX_SIZE, cfg)
    im_scales.append(im_scale)
    # processed_ims.append(im)

    # Create a blob to hold the input images
    blob = np.zeros((num_images, im.shape[0], im.shape[1], 3),
                    dtype=np.float32)
    blob[0, :, :, :] = im
    # blob = im_list_to_blob(processed_ims)

    return blob, im_scales


# def im_list_to_blob(ims):
#     """Convert a list of images into a network input.
#
#     Assumes images are already prepared (means subtracted, BGR order, ...).
#     """
#     max_shape = np.array([im.shape for im in ims]).max(axis=0)
#     num_images = len(ims)
#     blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
#                     dtype=np.float32)
#     for i in range(num_images):
#         im = ims[i]
#         blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
#
#     return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size, cfg):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    # if cfg.TRAIN.RANDOM_DOWNSAMPLE:
    #     r = 0.6 + np.random.rand() * 0.4
    #     im_scale *= r
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
