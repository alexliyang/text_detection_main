import pickle
import os
import numpy as np
import scipy.sparse


class roidb(object):
    def __init__(self, config=None):
        print('roidb initializing......')
        assert config != None, 'roidb lack config'

        self._image_path = config.TRAIN.TRAIN_PATH + '/Imageset'
        self._image_gt = config.TRAIN.TRAIN_PATH + '/Imageinfo'

        self._classes = ('__background__', 'text')
        self._num_classes = 2

        this._setup()




    def _setup(self):
        self._roidb = self._gt_roidb()
        self._image_index = self._load_image_set_index()

    @property
    def roidb(self):
        return self._roidb


    def _gt_roidb(self, config):
        cache_file = os.path.join(config.TRAIN.CACHE_PATH, 'roidb.pkl')
        gt_roidb = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_roidb = pickle.load(fid)
            print('gt roidb loaded from {}'.format(cache_file))

        else:
            gt_roidb = [self._process_each_image_gt(index)
                        for index in self._image_index]
            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb

    def _load_image_set_index(self):
        image_set_file = os.path.join(self.TRAIN_PATH, 'train_set.txt')
        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index


    def _process_each_image_gt(self, image_name):

        filename = os.path.join(self._image_gt, image_name + '.txt')
        with open(filename, 'w') as f:
            image_info = f.readlines()

        num_objs = len(gt_boxes)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self._num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # seg_areas = np.zeros((num_objs), dtype=np.float32)
        # ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, box in enumerate(gt_boxes):

            box = box.split(',')
            x1 = float(box[0])
            x2 = float(box[1])
            x3 = float(box[2])
            x4 = float(box[3])


            # diffc = obj.find('difficult')
            # difficult = 0 if diffc == None else int(diffc.text)
            # ishards[ix] = difficult

            # cls = self._class_to_ind[obj.find('name').text.lower().strip()]

            # 每个cls 初识都是1 因为都是正例
            cls = 1
            boxes[ix, :] = [x1, y1, x2, y2]
            # gt_classes[ix] = cls
            # overlaps[ix, cls] = 1.0
            # # seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {
                'image_name': image_name,
                'image_info': [],
                'boxes': boxes,
                'gt_classes': gt_classes,
                # 'gt_ishard': ishards,
                # 'gt_overlaps': overlaps,
                'flipped': False,
                # 'seg_areas': seg_areas
                }
    def append_flipped_images(self):
        pass


def get_training_roidb(config):
    print('Preparing training data...')

    base_roidb = roidb(config)

    # 是否要翻转图片，如果需要的话将翻转的图片追加
    if config.TRAIN.USE_FLIPPED:
        base_roidb.append_flipped_images()
    # base_roidb生成roidb属性
    base_roidb.prepare_roidb()

    print('done')
