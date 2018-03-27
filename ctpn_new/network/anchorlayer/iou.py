import numpy as np


def bbox_overlaps(anchors, gt_boxes):
    assert anchors.shape[1] == 4
    assert gt_boxes[1] == 4
    """
    Parameters
    ----------
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):         # 对query_boxex遍历
        box_area = (           # 每个query_boxex的面积
            (gt_boxes[k, 2] - gt_boxes[k, 0] + 1) *
            (gt_boxes[k, 3] - gt_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(anchors[n, 2], gt_boxes[k, 2]) -
                max(anchors[n, 0], gt_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(anchors[n, 3], gt_boxes[k, 3]) -
                    max(anchors[n, 1], gt_boxes[k, 1]) + 1
                )
                if ih > 0:
                    na = iw * ih  # 交集的面积

                    # 并集的面积
                    ua = float(
                        (anchors[n, 2] - anchors[n, 0] + 1) *
                        (anchors[n, 3] - anchors[n, 1] + 1) +
                        box_area - na
                    )
                    overlaps[n, k] = na / ua
    return overlaps


if __name__ == "__main__":
    from IPython import embed; embed()