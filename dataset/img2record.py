import os
import numpy as np
from PIL import Image
import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from tqdm import trange
import contextlib2
import pandas as pd

flags.DEFINE_string('dataset', '../dataset/train_5fold.csv', '训练集路径')
flags.DEFINE_string('train_record_path', '../dataset/record/train.record', '训练集存储路径')
flags.DEFINE_string('val_record_path', '../dataset/record/val.record', '验证集存储路径')
flags.DEFINE_string('test_record_path', '../dataset/record/test.record', '验证集存储路径')
flags.DEFINE_string('train_txt', '../dataset/train.txt', '训练集txt路径')
flags.DEFINE_string('val_txt', '../dataset/val.txt', '验证集txt路径')
flags.DEFINE_string('test_txt', '../dataset/test.txt', '验证集txt路径')


def create_tfrecord():
    df = pd.read_csv('../dataset/train_5fold.csv')  # 读取数据
    save_path = ['../dataset/record/train1.record', '../dataset/record/train2.record', '../dataset/record/train3.record',
                 '../dataset/record/train4.record', '../dataset/record/train5.record']
    for i in range(5):
        data = df[(df['fold'] == i)]
        data['image_id'] = '../dataset/train/' + data['image_id']
        data = data.sample(frac=1)
        create2records(np.array(data['image_id']), np.array(data['label']), save_path[i], 5)

    return 0


def image2tfrecord(image_list, label_list, filename):
    # 生成字符串型的属性
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 生成整数型的属性
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    print("len=", len(image_list))
    # 创建一个writer来写TFRecord文件，filename是输出TFRecord文件的地址

    writer = tf.io.TFRecordWriter(filename)
    len2 = len(image_list)
    for i in trange(len2):
        # 读取图片并解码
        image = Image.open(image_list[i])
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image = image.resize((256, 256))
        # 转化为原始字节
        image_bytes = image.tobytes()

        # 创建字典
        features = {}
        # 用bytes来存储image
        features['image_raw'] = _bytes_feature(image_bytes)
        # 用bytes来存储label
        features['label'] = _int64_feature(int(label_list[i]))
        # 将所有的feature合成features
        tf_features = tf.train.Features(feature=features)
        # 将样本转成Example Protocol Buffer，并将所有的信息写入这个数据结构
        tf_example = tf.train.Example(features=tf_features)
        # 序列化样本
        tf_serialized = tf_example.SerializeToString()
        # 将序列化的样本写入trfrecord
        writer.write(tf_serialized)
    writer.close()


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.
  Args:
    exit_stack: A context2.ExitStack used to automatically closed the TFRecords
      opened in this function.
    base_path: The base path for all shards
    num_shards: The number of shards
  Returns:
    The list of opened TFRecords. Position k in the list corresponds to shard k.
  """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def create2records(image_list, label_list, output_path, num_shards):
    # 生成字符串型的属性
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # 生成整数型的属性
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = open_sharded_output_tfrecords(
            tf_record_close_stack, output_path, num_shards)
        for idx, image in enumerate(image_list):
            if idx % 100 == 0:
                logging.info('On image %d of %d', idx, len(image_list))
            image = Image.open(image_list[idx])
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image = image.resize((800, 600))
            # 转化为原始字节
            image_bytes = image.tobytes()

            # 创建字典
            features = {}
            # 用bytes来存储image
            features['image_raw'] = _bytes_feature(image_bytes)
            # 用bytes来存储label
            features['label'] = _int64_feature(int(label_list[idx]))
            # 将所有的feature合成features
            tf_features = tf.train.Features(feature=features)
            # 将样本转成Example Protocol Buffer，并将所有的信息写入这个数据结构
            tf_example = tf.train.Example(features=tf_features)
            # 序列化样本
            tf_serialized = tf_example.SerializeToString()

            shard_idx = idx % num_shards
            output_tfrecords[shard_idx].write(tf_serialized)
        logging.info('Finished writing %d images', idx + 1)


def main(_grgv):
    create_tfrecord()

if __name__ == '__main__':
    app.run(main)
