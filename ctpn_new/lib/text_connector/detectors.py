#coding:utf-8
import numpy as np
from .text_proposal_connector import TextProposalConnector
from .text_proposal_connector_oriented import TextProposalConnector as TextProposalConnectorOriented


class TextDetector:
    def __init__(self, cfg):
        self._cfg = cfg
        # 测试模式选择
        self.mode = cfg.TEST.DETECT_MODE
        # 检测水平的
        if self.mode == "H":
            self.text_proposal_connector = TextProposalConnector(cfg)
        # 检测倾斜的
        elif self.mode == "O":
            self.text_proposal_connector = TextProposalConnectorOriented(cfg)

    def detect(self, text_proposals, scores, size):
        # 删除得分较低（小于0.7）的proposal
        # keep_inds = np.where(scores > self._cfg.TEXT_PROPOSALS_MIN_SCORE)[0]
        # text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 按得分排序，在网络里面已经排序了
        # sorted_indices = np.argsort(scores.ravel())[::-1]
        # text_proposals, scores = text_proposals[sorted_indices], scores[sorted_indices]

        # 怎么又做nms？？？？？？？？？？？？？在输出的时候已经非极大值抑制了
        # keep_inds = nms(np.hstack((text_proposals, scores)), self._cfg.TEXT_PROPOSALS_NMS_THRESH)
        # text_proposals, scores = text_proposals[keep_inds], scores[keep_inds]

        # 获取检测结果。返回一个N×9的矩阵，表示N个拼接以后的完整的文本框。
        # 每一行，前八个元素一次是左上，右上，左下，右下的坐标，最后一个元素是文本框的分数
        text_recs = self.text_proposal_connector.get_text_lines(text_proposals, scores, size)
        keep_inds = self.filter_boxes(text_recs)
        return text_recs[keep_inds]

    def filter_boxes(self, boxes):
        heights = np.zeros((len(boxes), 1), np.float)
        widths = np.zeros((len(boxes), 1), np.float)
        scores = np.zeros((len(boxes), 1), np.float)
        index = 0
        for box in boxes:
            # 高度
            heights[index] = abs(box[5]-box[1])+1
            widths[index] = abs(box[2]-box[0])+1
            scores[index] = box[8]
            index += 1

        return np.where((widths/heights > self._cfg.MIN_RATIO) & (scores > self._cfg.LINE_MIN_SCORE) &
                        (widths > (self._cfg.TEXT_PROPOSALS_WIDTH*self._cfg.MIN_NUM_PROPOSALS)))[0]
