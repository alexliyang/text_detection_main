import pickle
import os
import numpy as np
import scipy.sparse
import PIL.Image


def get_training_roidb(config):
    print('Preparing training data...')

    base_roidb = roidb(config)

    # 是否要翻转图片，如果需要的话将翻转的图片追加
    if config.TRAIN.USE_FLIPPED:
        base_roidb.append_flipped_images()

    print('done')


class roidb(object):
    def __init__(self, config=None):
        print('roidb initializing......')
        assert config != None, 'roidb lack config'

        self.config = config
        self._image_path = config.TRAIN.TRAIN_PATH + '/Imageset'
        self._image_gt = config.TRAIN.TRAIN_PATH + '/Imageinfo'
        self._train_data_path = config.TRAIN.TRAIN_PATH

        self._classes = ('__background__', 'text')
        self._num_classes = 2

        # self._image_index =[]
        self._setup()

    '''_image_index ['xxx.jpg','yyy.jpg'......]'''

    def _setup(self):
        self._load_image_set_index()

        self._gt_roidb()

    @property
    def roidb(self):
        return self._roidb

    def _get_image_path_with_name(self, image_name):
        image_path = os.path.join(self._image_path, image_name)
        assert os.path.exists(image_path), \
            'Image does not exist: {}'.format(image_path)
        return image_path

    '''train_set.txt
       xxxxx.jpg, width, height, channel, scale
    '''

    def _load_image_set_index(self):
        image_set_file = os.path.join(self._train_data_path, 'train_set.txt')

        assert os.path.exists(image_set_file), 'Path does not exist: {}'.format(image_set_file)
        image_index = []
        image_info = []
        with open(image_set_file) as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip().split(',')
            image_index.append(line[0])  # xxxx.name
            image_info.append(line[1:])

        self._image_index = image_index
        self._image_info = image_info

    def _gt_roidb(self):
        cache_file = os.path.join(self.config.TRAIN.CACHE_PATH, 'roidb.pkl')

        print(cache_file)
        gt_roidb = None
        if os.path.exists(cache_file) and self.config.TRAIN.USE_CACHED:
            with open(cache_file, 'rb') as fid:
                gt_roidb = pickle.load(fid)
            print('gt roidb loaded from {}'.format(cache_file))

        else:
            print(self._image_index)
            gt_roidb = [self._process_each_image_gt(index, image_name)
                        for index, image_name in enumerate(self._image_index)]
            # if not os.path.exists(cache_file):
            #     os.makedirs(cache_file)

            with open(cache_file, 'wb') as fid:
                pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)

            print('wrote gt roidb to {}'.format(cache_file))
        self._roidb = gt_roidb

    def _process_each_image_gt(self, index, image_name):

        filename = os.path.join(self._image_gt, image_name + '.txt')
        print(filename)
        with open(filename, 'r') as f:
            gt_boxes = f.readlines()

        num_objs = len(gt_boxes)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self._num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        # seg_areas = np.zeros((num_objs), dtype=np.float32)
        # ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        # size = PIL.Image.open(self._get_image_path_with_name(image_name)).size
        single_img_info = self._image_info[index]
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
            'image_path': self._get_image_path_with_name(image_name),
            'image_name': image_name,
            'width': single_img_info[0],
            'height': single_img_info[1],
            'image_scale': single_img_info[2],
            'boxes': boxes,
            # 'gt_classes': gt_classes,
            # 'gt_ishard': ishards,
            # 'gt_overlaps': overlaps,
            # 'flipped': False,
            # 'seg_areas': seg_areas
        }

    def append_flipped_images(self):
        pass

    def add_bbox_regression_targets(self):
        pass

    def _compute_targets(self):
        pass

#
# if __name__ == 'main':
#     d = roidb()
