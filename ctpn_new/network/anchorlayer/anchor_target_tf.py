# -*- coding:utf-8 -*-
import numpy as np
import numpy.random as npr
from .generate_anchors import generate_anchors
from .iou import bbox_overlaps, bbox_intersections
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform


def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride = (16,), anchor_scales = (16,)):

    # 生成基本的anchor,一共10个,返回一个10行4列矩阵，每行为一个anchor，返回的只是基于中心的相对坐标
    _anchors = generate_anchors()
    _num_anchors = _anchors.shape[0]  # 10个anchor

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # 第一张图片的im_info为[[800, 600, 1]]，所以要有下面这句话
    im_info = im_info[0]  # 图像的高宽及通道数

    assert rpn_cls_score.shape[0] == 1, 'Only single item batches are supported'

    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]  # feature-map的高宽

    # ====================================================
    shift_x = np.arange(0, width) * _feat_stride  # 返回一个列表，[0, 16, 32, 48, ...]
    shift_y = np.arange(0, height) * _feat_stride

    # 此时，shift_x作为一个行向量往下复制， 复制的次数等于shift_y的长度
    # 而shift_y作为一个列向量朝右复制，复制的次数等于shift_x的长度。这样他们的维度完全相同
    shift_x, shift_y = np.meshgrid(shift_x, shift_y) # in W H order
    # K is H x W
    # .ravel()将数组按行展开，展开为一行
    # .vstack()将四个展开列的以为数组垂直堆叠起来，再转置
    # shift的行数为像素个数，列数为4
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors  # 10个anchor
    K = shifts.shape[0]  # feature-map的像素个数

    # 前者的shape为(1, 10, 4), 后者的shape为(像素数, 1, 4)两者相加
    # 结果为(像素数, 10, 4) python数组广播相加。。。。。。。有待理解
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))

    # 至此，每一行为一个anchor， 每十行为一个滑动窗对应的十个anchor，第二个十行为往右走所对应的十个anchors
    all_anchors = all_anchors.reshape((K * A, 4))
    # ================================================== 至此，anchors生成好了

    # 总的anchors数目为滑动窗个数乘以10
    # total_anchors = int(K * A)

    # 仅保留那些还在图像内部的anchor
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]
    total_valid_anchors = len(inds_inside)

    # 只保留内部的anchors
    # 经过验证，这里的anchors的宽度全部是16
    anchors = all_anchors[inds_inside, :]  # 保留那些在图像内的anchor

    # 至此，anchor准备好了
    # ===============================================================================
    # --------------------------------------------------------------
    # label: 1 is positive, 0 is negative, -1 is dont care
    # (A)
    labels = np.empty((total_valid_anchors, ), dtype=np.int8)
    labels.fill(-1)  # 初始化label，均为-1

    # 计算anchor和gt-box的overlap，用来给anchor上标签
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))  # 假设anchors有x个，gt_boxes有y个，返回的是一个（x,y）的数组

    # argmax_overlaps[0]表示第0号anchor与所有GT的IOU最大值的脚标
    argmax_overlaps = overlaps.argmax(axis=1)

    # 返回一个一维数组，第i号元素的值表示第i个anchor与最可能的GT之间的IOU
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]

    # gt_argmax_overlaps[0]表示所有anchor中与第0号GT的IOU最大的那个anchor
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # 返回一个一维数组，第i号元素的值表示第i个GT与所有anchor的IOU最大的那个值
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]

    #  ?????????干啥的？[2, 2, 4, 5]表示2号anchor与所有的GT有两个最大值， 4号anchor与所有的GT有一个最大值
    # 这里的最大值，是指定一个GT后，与所有anchor的最大值
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

    # 这里cfg.TRAIN.RPN_CLOBBER_POSITIVES=False
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:   # 这里要执行

        # 这里cfg.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
        # 对于某个anchor，他与所有GT的最大IOU值都小于0.3, 那定义为背景
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # 由于所有的anchors是穷举扫描，覆盖了全部的图片，对于某个GT，与其有最大的IOU的一定是文字
    labels[gt_argmax_overlaps] = 1

    # cfg.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1  # overlap大于0.7的认为是前景

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:  # 这里是False
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # preclude dontcare areas
    # if dontcare_areas is not None and dontcare_areas.shape[0] > 0:  # 这里我们暂时不考虑有doncare_area的存在
    #     # intersec shape is D x A
    #     intersecs = bbox_intersections(
    #         np.ascontiguousarray(dontcare_areas, dtype=np.float), # D x 4
    #         np.ascontiguousarray(anchors, dtype=np.float) # A x 4
    #     )
    #     intersecs_ = intersecs.sum(axis=0) # A x 1
    #     labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    #这里我们暂时不考虑难样本的问题
    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    # if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
    #     assert gt_ishard.shape[0] == gt_boxes.shape[0]
    #     gt_ishard = gt_ishard.astype(int)
    #     gt_hardboxes = gt_boxes[gt_ishard == 1, :]
    #     if gt_hardboxes.shape[0] > 0:
    #         # H x A
    #         hard_overlaps = bbox_overlaps(
    #             np.ascontiguousarray(gt_hardboxes, dtype=np.float), # H x 4
    #             np.ascontiguousarray(anchors, dtype=np.float)) # A x 4
    #         hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
    #         labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
    #         max_intersec_label_inds = hard_overlaps.argmax(axis=1) # H x 1
    #         labels[max_intersec_label_inds] = -1 #

    # subsample positive labels if we have too many
    # 对正样本进行采样，如果正样本的数量太多的话
    # TODO 限制正样本的数量不超过150个
    # TODO 这个后期可能还需要修改，毕竟如果使用的是字符的片段，那个正样本的数量是很多的。
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 0.5*300
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)  # 随机去除掉一些正样本
        labels[disable_inds] = -1  # 变为-1

    # subsample negative labels if we have too many
    # 对负样本进行采样，如果负样本的数量太多的话
    # 正负样本总数是300，限制正样本数目最多150，
    # 如果正样本数量小于150，差的那些就用负样本补上，凑齐256个样本
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    # 至此， 上好标签，开始计算rpn-box的真值
    # --------------------------------------------------------------
    # bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    # 根据anchor和gtbox计算得真值（anchor和gtbox之间的偏差）
    # 输入是所有的anchors，以及与之IOU最大的那个GT，返回是一个N×4的矩阵，每行表示一个anchor与对应的IOU最大的GT的x,y,w,h回归
    """这里对所有的anchor都计算了回归，事实上只需要对正例进行回归即可"""
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

    # RPN_BBOX_INSIDE_WEIGHTS=[0, 1, 0, 1]
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)

    # 暂时使用uniform 权重，也就是正样本是1，负样本是0,
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:  # cfg.TRAIN.RPN_POSITIVE_WEIGHT = -1
        # uniform weighting of examples (given non-uniform sampling)
        num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights  # 外部权重，前景是1，背景是0
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print('means:')
        print(means)
        print('stdevs:')
        print(stds)

    # map up to original set of anchors
    # 一开始是将超出图像范围的anchor直接丢掉的，现在在加回来， 加回来的的标签全部置为-1
    # labels是内部anchor的分类， total_anchors是总的anchor数目， inds_inside是内部anchor的索引
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)  # 这些anchor的label是-1，也即dontcare
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)  # 这些anchor的真值是0，也即没有值
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)  # 内部权重以0填充
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)  # 外部权重以0填充

    if DEBUG:
        print('rpn: max max_overlap', np.max(max_overlaps))
        print('rpn: num_positive', np.sum(labels == 1))
        print('rpn: num_negative', np.sum(labels == 0))
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print('rpn: num_positive avg', _fg_sum / _count)
        print('rpn: num_negative avg', _bg_sum / _count)

    # labels
    labels = labels.reshape((1, height, width, A))  # reshape一下label
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))#reshape

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


    # data是内部anchor的分类， count是总的anchor数目， inds是内部anchor的索引
def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5
    """
    到这里为止， 用下面的代码验证了 ex_rois的宽度全部为16
    mywidth = ex_rois[:, 2]-ex_rois[:, 0] + 1
    for i in mywidth:
        if i != 16:
            print("=============++++++++++++++===========", mywidth)
    """
    # bbox_transform函数的输入是anchors， 和GT的坐标部分
    # 输出是一个N×4的矩阵，每行表示一个anchor与对应的IOU最大的GT的x,y,w,h回归
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
