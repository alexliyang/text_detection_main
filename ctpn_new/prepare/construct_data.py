"""
这个脚本用来处理原始的数据，将图片按照短边600为标准进行缩放，如果缩放后长边超过1200，按照长边1200缩放，同时要缩放坐标
要将数据整理成的格式如下，存放在dataset/for_train下
以dataset/for_train为根目录
---| Imageset 保存图片文件
   | Imageinfo 保存每张图片对应的txt文本
   ----|xxxxxx.txt xxxxxx为图片名(不带扩展名)，每一行为一个文本框，格式为xmin,ymin,xmax,ymax
   ----|..........
   | train_set.txt 保存所有训练样本对应的文件名，每个占一行，格式位xxxx.jpg,width,height,channel,scale （scale为缩放比例）
### 要求在ctpn_new 目录下能够直接运行，直接读取 dataset/ICPR_text_train 下的原始训练数据进行处理，结果直接输出到到dataset下，不可人工复制粘贴，
ctpn_new/dataset 目录下的所有文件将被git忽略，以提高push速度和减少不必要的文件冲突###
"""
"""原始训练集图像存放在/dataset/ICPR_text_train/image目录下
原始训练集txt存放在/dataset/ICPR_text_train/text目录下
由原始数据集得到训练数据集调用rawdata2traindata()
"""

import os
from PIL import Image

image_dir = "./dataset/ICPR_text_train/image"  # 原始训练数据集图像目录
txt_dir = "./dataset/ICPR_text_train/text"   # 原始训练数据集txt文本目录
txtfortrain_dir = "./dataset/for_train/Imageinfo" # 保存每张图片对应的txt文本的目录
imagefortain_dir = "./dataset/for_train/Imageset" # 保存图片文件的目录

if not os.path.exists(txtfortrain_dir):
    os.makedirs(txtfortrain_dir)
if not os.path.exists(imagefortain_dir):
    os.makedirs(imagefortain_dir)

def rawdata2traindata(config):
    # 将所有训练样本对应的文件名保存在dataset/for_train/train_set.txt 中，每个占一行,格式为xxxx.jpg, width, height, channel, scale
    # 保存图片文件，将图片按照短边600为标准进行缩放，如果缩放后长边超过1200，按照长边1200缩放，同时要缩放坐标
    # 保存每张图片对应的txt文本，每一行为一个文本框，格式为xmin,ymin,xmax,ymax
    imagedata_process(config)

def imagedata_process(config):
    filename = "train_set.txt"
    pathdir = "./dataset/for_train"
   #判断train_set.txt是否存在，存在则删除
    if os.path.exists(pathdir + '/' + filename):
        os.remove(pathdir + '/' + filename)
    # 创建文件train_set.txt
    trainsetfile = open(pathdir + '/' + filename, 'w')
    image_files = os.listdir(image_dir)  # 得到文件夹下的所有文件名称
    for image_filename in image_files:
        #print(image_filename)
        imagename, ext = os.path.splitext(image_filename)        #分离文件名和扩展名
        rawImage = Image.open(image_dir + "/" + image_filename, 'r').convert('RGB')  # 打开原始图片，统一转换为RGB
        imgSize = rawImage.size      # 图片的高（行数）和宽（列数）
        img_height = imgSize[0]     # 图片的高（行数）
        img_width = imgSize[1]      # 图片的宽（列数）
        #print(imgSize)
        #print(img_height)
        #print(img_width)
        # 图片按照短边600为标准进行缩放,如果缩放后长边超过1200，按照长边1200缩放
        if img_width <= img_height:
            width = int(600)
            height = int(float(img_height) * 600 / float(img_width))
            if height > 1200:
                height = int(1200)
                width = int(float(img_width) * 1200 / float(img_height))
                txtdata_process(imagename, 1200, img_height, config, width, height)
                scale = round(1200 / float(img_height), 2)
            else:
                txtdata_process(imagename, 600, img_width, config, width, height)
                scale = round(600 / float(img_width), 2)
        else:
            height = int(600)
            width = int(float(img_width) * 600 / float(img_height))
            if width > 1200:
                width = int(1200)
                height = int(float(img_height) * 1200 / float(img_width))
                txtdata_process(imagename, 1200, img_width, config, width, height)
                scale = round(1200 / float(img_width), 2)
            else:
                txtdata_process(imagename, 600, img_height, config, width, height)
                scale = round(600 / float(img_height), 2)
        trainImage = rawImage.resize((height, width), Image.ANTIALIAS) #图片缩放
        trainSize = trainImage.size  # 图片的高（行数）和宽（列数）
        train_height = trainSize[0]  # 图片的高（行数）
        train_width = trainSize[1]  # 图片的宽（列数）
        #print(trainSize)
        #print(train_height)
        #print(train_width)
        # 如果文件已存在则删除
        if os.path.exists(imagefortain_dir + '/' + image_filename):
            os.remove(imagefortain_dir + '/' + image_filename)
        trainImage.save(imagefortain_dir + '/' + image_filename)        #保存缩放后的训练图像
        trainsetfile.write(image_filename + "," + str(train_width) + "," + str(train_height) + "," + str(3) + "," + str(scale)+ "\n")
    trainsetfile.close()

def txtdata_process(file, numerator, denominator, config, width, height):
    if os.path.exists(txtfortrain_dir + '/' + file + ".txt"):
        os.remove(txtfortrain_dir + '/' + file + ".txt")
    # 创建用于训练的txt文件
    fortraintxtfile = open(txtfortrain_dir + '/' + file + ".txt", 'a')
    f = open(txt_dir + "/" + file + ".txt", 'r', encoding='UTF-8')  # 打开原始txt文件
    iter_f = iter(f)  # 创建迭代器
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        tmp = line.split(",")  # 将原始行以“，”分割
        #print(tmp)
        # 判断倾角，小于某阈值舍弃
        threshold = config.PREPARE.SLOP_THRESHOLD   #阈值
        if not float(tmp[0]) == float(tmp[2]):   #X1和X2不相等，即文本框是倾斜的
            y = max(float(tmp[1]), float(tmp[3])) - min(float(tmp[1]), float(tmp[3]))
            x = max(float(tmp[0]), float(tmp[2])) - min(float(tmp[0]), float(tmp[2]))
            if y/x <= threshold:
                continue
        xmin = round(min(float(tmp[0]), float(tmp[2]), float(tmp[4]), float(tmp[6]))*float(numerator)/float(denominator), 2)
        ymin = round(min(float(tmp[1]), float(tmp[3]), float(tmp[5]), float(tmp[7]))*float(numerator)/float(denominator), 2)
        xmax = round(max(float(tmp[0]), float(tmp[2]), float(tmp[4]), float(tmp[6]))*float(numerator)/float(denominator), 2)
        ymax = round(max(float(tmp[1]), float(tmp[3]), float(tmp[5]), float(tmp[7]))*float(numerator)/float(denominator), 2)
        if testGT(xmin, ymin, xmax, ymax, width, height):
            fortraintxtfile.write(str(xmin) + "," + str(ymin) + "," + str(xmax) + "," + str(ymax) + "\n")
    fortraintxtfile.close()

def testGT(xmin, ymin, xmax, ymax, width, height):
    """
    判断GT是否在图像范围内
    """
    if xmin < 0 or xmin > width:
        return False
    if xmax < 0 or xmax > width:
        return False
    if ymin < 0 or ymin > height:
        return False
    if ymax < 0 or ymax > height:
        return False
    return True

import sys
sys.path.append(os.getcwd())
from lib.load_config import load_config
cfg = load_config()
rawdata2traindata(cfg)
