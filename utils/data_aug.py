import tensorflow as tf
from absl import logging, flags, app
from absl.flags import FLAGS
import numpy as np

from absl.flags import FLAGS
from utils.imaug.GridMask import gridmask
from utils.imaug.CutMix import cutmix
from utils.imaug.geometry import random_zoom
from utils.imaug.colorjitter import adjust_color
from utils.imaug.Cutout import cutout
from utils.imaug.RnadomErasing import random_erasing
from utils.imaug.Mixup import mixup
from utils.imaug.RandomFlip import random_flip

flags.DEFINE_integer('num_train_pic', 1732, '训练集图片数')
flags.DEFINE_integer('num_val_pic', 300, '验证集集图片数')


# tfrecord格式转换
def _parse_tfrecord(tfrecord, cfg):
    x = tf.io.parse_single_example(tfrecord,  features={
                # tf.FixedLenFeature解析的结果为一个tensor
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64),
                })  # 取出包含image和label的feature对象

    # tf.decode_raw可以将字符串解析成图像对应的像素数组
    image = tf.io.decode_raw(x['image_raw'], tf.uint8)
    # 根据图像尺寸，还原图像
    image = tf.reshape(image, [cfg.DATASET.RAW_SIZE[0], cfg.DATASET.RAW_SIZE[1], cfg.DATASET.CHANNELS])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, cfg.DATASET.SIZE)
    label = tf.cast(x['label'], tf.int32)

    return image, label


# 数据读取
def get_dataset(file, cfg, is_training=False):
    # 获取文件名列表
    files = tf.data.Dataset.list_files(file)
    # 读入原始的TFRecord文件, 获得一个 tf.data.Dataset 数据集对象
    dataset = files.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda x: _parse_tfrecord(x, cfg))

    if is_training:
        dataset = dataset.shuffle(buffer_size=300)
        # dataset = dataset.map(adjust_color)
        # dataset = dataset.map(lambda x, y: random_flip(x, y, vertical=False))
        # dataset = dataset.map(random_zoom)
        # dataset = dataset.map(gridmask)
        # dataset = dataset.map(lambda x, y: random_erasing(x, y, probability=0.5, sl=0.02, sh=0.3, r1=0.3))
        dataset = dataset.batch(cfg.TRAIN.BATCH_SIZE)
        # dataset = dataset.map(lambda x, y: cutmix(x, y, batch_size, classes))
        # dataset = dataset.map(lambda x, y: mixup(x, y, batch_size, classes, probability=0.5))
        # dataset = dataset.map(lambda x, y: cutout(x, y, mask_size=80))
    else:
        dataset = dataset.batch(1)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset
