import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, average_precision_score


class ClassMetric(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.y_true = []
        self.y_pred = []

    def f1score(self):
        y_true = tf.cast(self.y_true, dtype=tf.int32)
        y_pred = tf.cast(self.y_pred, dtype=tf.int32)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')

        return f1

    def acc(self):
        y_true = tf.cast(self.y_true, dtype=tf.int32)
        y_pred = tf.cast(self.y_pred, dtype=tf.int32)
        acc = np.sum(y_pred == y_true) / len(y_true)

        return acc

    def mAP(self):
        y_true = tf.cast(self.y_true, dtype=tf.int32)
        y_pred = tf.cast(self.y_pred, dtype=tf.int32)
        y_true = np.array(tf.one_hot(y_true, self.num_class))
        y_pred = np.array(tf.one_hot(y_pred, self.num_class))
        mAP = average_precision_score(y_true, y_pred)

        return mAP

    def addBatch(self, label, predict):
        assert predict.shape == label.shape
        self.y_true.extend(label)
        self.y_pred.extend(predict)

    def reset(self):
        self.y_true = []
        self.y_pred = []


# 计算参数量
def param_count():
    total_parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


# 计算FLOPs
def count_flops(graph):
    flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


