#coding:utf-8
import tensorflow as tf
import numpy as np
import shutil
import glob
import os, cv2
from lib.timer import Timer
from lib.utils.test_util import test_util
from lib.text_connector.detectors import TextDetector

class TestClass(object):
    def __init__(self,cfg,network):
        self._cfg = cfg
        self._net = network
        self.tu = test_util(cfg)
        #self.setup()

     # 画方框,被ctpn()调用
    def draw_boxes(self,img, image_name, boxes, scale):
        base_name = image_name.split('/')[-1]
        with open(self._cfg.TEST.RESULT_DIR+'/' + 'res_{}.txt'.format(base_name.split('.')[0]), 'w') as f:
            for box in boxes:
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                if box[8] >= 0.9:
                    color = (0, 255, 0)
                elif box[8] >= 0.8:
                    color = (255, 0, 0)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

                min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
                max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
                max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))

                line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
                f.write(line)

        img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(self._cfg.TEST.RESULT_DIR, base_name), img)

    #改变图片的尺寸，被ctpn()调用
    def resize_im(self,im, scale, max_scale=None):
        f = float(scale) / min(im.shape[0], im.shape[1])
        if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
            f = float(max_scale) / max(im.shape[0], im.shape[1])
        return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

    #被test_net()调用
    def ctpn(self,sess, net, image_name):
        timer = Timer()
        timer.tic()

        img = cv2.imread(image_name)
        #resize_im的传入参数根据配置文件
        img, scale = self.resize_im(img, scale=self._cfg.SCALE, max_scale=self._cfg.MAX_SCALE)
        scores, boxes = self.tu.test_ctpn(sess, net, img)

        #此处调用了一个文本检测器，未实现
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        self.draw_boxes(img, image_name, boxes, scale)
        timer.toc()
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    def test_net(self):
        #定义结果输出的路径(trd 是test_result_dir缩写 )
        trd= self._cfg.TEST.RESULT_DIR
        if os.path.exists(trd):
            shutil.rmtree(trd)
        os.makedirs(trd)

        #创建一个Session
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        #获取一个Saver()实例
        saver = tf.train.Saver()

        #恢复模型参数
        try:
            ckpt = tf.train.get_checkpoint_state(self._cfg.COMMON.CHKPT)
            print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('done')
        except:
            raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

        #加载图片
        im_names = glob.glob(os.path.join(self._cfg.TEST.DATA_DIR, 'for_test', '*.png')) + \
                   glob.glob(os.path.join(self._cfg.TEST.DATA_DIR, 'for_test', '*.jpg'))

        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(('Testing for image {:s}'.format(im_name)))
            self.ctpn(sess, self._net, im_name)

