import os
from os.path import join
import cv2
import numpy as np


def draw_boxes(img, image_name, boxes, scale):
    base_name = image_name

    for box in boxes:
        # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
        #     continue
        # if box[8] >= 0.9:
        #     color = (0, 255, 0)
        # elif box[8] >= 0.8:
        #     color = (255, 0, 0)
        color = (0,255, 0)

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[2]), int(box[3])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[6]), int(box[7])), color, 2)

        # min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        # min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        # max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        # max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        #
        # line = ','.join([str(min_x), str(min_y), str(max_x), str(max_y)]) + '\r\n'
        # f.write(line)

    img = cv2.resize(img, None, None, fx=1.0 / 1, fy=1.0 / 1, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("analysis/tmp", base_name + '.jpg', ), img)


def fix_coordinate(x, mi, mx):
    x = x.split('.')[0]
    x = int(x)
    if x < mi:
        return mi
    if x > mx:
        return mx
    return x


def main():
    dataset_path = 'dataset/ICPR_text_train/text'
    image_path = 'dataset/ICPR_text_train/image'

    img_list = os.listdir(dataset_path)
    samples = []
    print(len(img_list))
    for img_path in img_list[:1000]:
        with open(join(dataset_path, img_path)) as f:
            lines = f.readlines()
        print(img_path)
        img = cv2.imread(join(image_path, os.path.splitext(img_path)[0] + '.jpg'))
        boxes = []
        try:
            len(img)
        except:
            continue
        mx = max(img.shape[0], img.shape[2])
        for line in lines:
            box = line.split(',')[:8]

            box = list(map(lambda x: fix_coordinate(x, 0, mx), box))
            boxes.append(box)
            # print(box)
            # print(boxes)
            # points = list(zip(line[0::2], line[1::2]))
            # print(points)
        draw_boxes(img, img_path, boxes, 1)


if __name__ == '__main__':
    main()
