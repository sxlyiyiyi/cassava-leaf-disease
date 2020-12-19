import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import UpSampling2D, add, concatenate, MaxPool2D, Dropout
import tensorflow.keras.backend as K
import numpy as np


def basic_Block(inputs, out_filters, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2D(out_filters, 3, padding='same', strides=strides, use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3,)(x)
    x = Activation('relu')(x)
    x = Conv2D(out_filters, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(input)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, inputs])

    x = Activation('relu')(x)
    return x


def bottleneck_Block(inputs, out_filters, strides=(1, 1), with_conv_shortcut=False):
    expansion = 4
    de_filters = int(out_filters / expansion)

    x = Conv2D(de_filters, 1, use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(de_filters, 3, strides=strides, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = Conv2D(out_filters, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)

    if with_conv_shortcut:
        residual = Conv2D(out_filters, 1, strides=strides, use_bias=False, kernel_initializer='he_normal')(inputs)
        residual = BatchNormalization(axis=3)(residual)
        x = add([x, residual])
    else:
        x = add([x, inputs])

    x = Activation('relu')(x)
    return x


# 第一个block, 包括两个3*3的下采样用于图片的输入和 N11
def stem_net(inputs):
    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=3)(x)
    # x = Activation('relu')(x)

    x = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = bottleneck_Block(x, 256, with_conv_shortcut=True)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)
    x = bottleneck_Block(x, 256, with_conv_shortcut=False)

    return x


# 第一个
def transition_layer1(x, out_chan):
    x0 = Conv2D(out_chan[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x0 = BatchNormalization(axis=3)(x0)
    x0 = Activation('relu')(x0)

    x1 = Conv2D(out_chan[1], 3, strides=(2, 2),
                padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = Activation('relu')(x1)

    return [x0, x1]


# block1_0
def make_branch1(x, out_chan):
    x1_0 = basic_Block(x[0], out_chan[0], with_conv_shortcut=False)
    x1_0 = basic_Block(x1_0, out_chan[0], with_conv_shortcut=False)
    x1_0 = basic_Block(x1_0, out_chan[0], with_conv_shortcut=False)
    x1_0 = basic_Block(x1_0, out_chan[0], with_conv_shortcut=False)

    x1_1 = basic_Block(x[1], out_chan[1], with_conv_shortcut=False)
    x1_1 = basic_Block(x1_1, out_chan[1], with_conv_shortcut=False)
    x1_1 = basic_Block(x1_1, out_chan[1], with_conv_shortcut=False)
    x1_1 = basic_Block(x1_1, out_chan[1], with_conv_shortcut=False)

    return [x1_0, x1_1]


# 不同分辨率之间的交互
def fuse_layer1(x, out_filters):
    # x0_0 = x[0]
    x0_1 = Conv2D(out_filters[0], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_1 = tf.compat.v1.image.resize_bilinear(x0_1, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0 = add([x[0], x0_1])
    x0 = Activation('relu')(x0)

    x1_0 = Conv2D(out_filters[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    # x1_1 = x[1]
    x1 = add([x1_0, x[1]])
    x1 = Activation('relu')(x1)
    return [x0, x1]


def transition_layer2(x, out_chan):
    # x0 = x[0]
    # x1 = x[1]
    x2 = Conv2D(out_chan[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2 = BatchNormalization(axis=3)(x2)
    x2 = Activation('relu')(x2)

    return [x[0], x[1], x2]


def make_branch2(x, out_filters):
    x2_0 = basic_Block(x[0], out_filters[0], with_conv_shortcut=False)
    x2_0 = basic_Block(x2_0, out_filters[0], with_conv_shortcut=False)
    x2_0 = basic_Block(x2_0, out_filters[0], with_conv_shortcut=False)
    x2_0 = basic_Block(x2_0, out_filters[0], with_conv_shortcut=False)

    x2_1 = basic_Block(x[1], out_filters[1], with_conv_shortcut=False)
    x2_1 = basic_Block(x2_1, out_filters[1], with_conv_shortcut=False)
    x2_1 = basic_Block(x2_1, out_filters[1], with_conv_shortcut=False)
    x2_1 = basic_Block(x2_1, out_filters[1], with_conv_shortcut=False)

    x2_2 = basic_Block(x[2], out_filters[2], with_conv_shortcut=False)
    x2_2 = basic_Block(x2_2, out_filters[2], with_conv_shortcut=False)
    x2_2 = basic_Block(x2_2, out_filters[2], with_conv_shortcut=False)
    x2_2 = basic_Block(x2_2, out_filters[2], with_conv_shortcut=False)

    return [x2_0, x2_1, x2_2]


def fuse_layer2(x, out_chan):
    x0_1 = Conv2D(out_chan[0], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_2 = Conv2D(out_chan[0], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)

    x0_1 = tf.compat.v1.image.resize_bilinear(x0_1, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0_2 = tf.compat.v1.image.resize_bilinear(x0_2, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0 = add([x[0], x0_1, x0_2])
    x0 = Activation('relu')(x0)

    x1_0 = Conv2D(out_chan[1], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization(axis=3)(x1_0)
    x1_2 = Conv2D(out_chan[1], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization(axis=3)(x1_2)

    x1_2 = tf.compat.v1.image.resize_bilinear(x1_2, [tf.shape(x[1])[1], tf.shape(x[1])[2]], align_corners=True)
    x1 = add([x1_0, x[1], x1_2])
    x1 = Activation('relu')(x1)

    x2_0 = Conv2D(out_chan[0], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_0 = Conv2D(out_chan[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization(axis=3)(x2_0)
    x2_1 = Conv2D(out_chan[2], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization(axis=3)(x2_1)

    x2 = add([x2_0, x2_1, x[2]])
    x2 = Activation('relu')(x2)

    return [x0, x1, x2]


# 变换通道数
def transition_layer3(x, out_chan):
    # x0 = x[0]
    # x1 = x[1]
    # x2 = x[2]

    x3 = Conv2D(out_chan[3], 3, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3 = BatchNormalization(axis=3)(x3)
    x3 = Activation('relu')(x3)

    return [x[0], x[1], x[2], x3]


def make_branch3(x, out_chan):
    x3_0 = basic_Block(x[0], out_chan[0], with_conv_shortcut=False)
    x3_0 = basic_Block(x3_0, out_chan[0], with_conv_shortcut=False)
    x3_0 = basic_Block(x3_0, out_chan[0], with_conv_shortcut=False)
    x3_0 = basic_Block(x3_0, out_chan[0], with_conv_shortcut=False)

    x3_1 = basic_Block(x[1], out_chan[1], with_conv_shortcut=False)
    x3_1 = basic_Block(x3_1, out_chan[1], with_conv_shortcut=False)
    x3_1 = basic_Block(x3_1, out_chan[1], with_conv_shortcut=False)
    x3_1 = basic_Block(x3_1, out_chan[1], with_conv_shortcut=False)

    x3_2 = basic_Block(x[2], out_chan[2], with_conv_shortcut=False)
    x3_2 = basic_Block(x3_2, out_chan[2], with_conv_shortcut=False)
    x3_2 = basic_Block(x3_2, out_chan[2], with_conv_shortcut=False)
    x3_2 = basic_Block(x3_2, out_chan[2], with_conv_shortcut=False)

    x3_3 = basic_Block(x[3], out_chan[3], with_conv_shortcut=False)
    x3_3 = basic_Block(x3_3, out_chan[3], with_conv_shortcut=False)
    x3_3 = basic_Block(x3_3, out_chan[3], with_conv_shortcut=False)
    x3_3 = basic_Block(x3_3, out_chan[3], with_conv_shortcut=False)

    return [x3_0, x3_1, x3_2, x3_3]


def fuse_layer3(x, num_chan):
    x0_1 = Conv2D(num_chan[0], 1, use_bias=False, kernel_initializer='he_normal')(x[1])
    x0_1 = BatchNormalization(axis=3)(x0_1)
    x0_2 = Conv2D(num_chan[0], 1, use_bias=False, kernel_initializer='he_normal')(x[2])
    x0_2 = BatchNormalization(axis=3)(x0_2)
    x0_3 = Conv2D(num_chan[0], 1, use_bias=False, kernel_initializer='he_normal')(x[3])
    x0_3 = BatchNormalization(axis=3)(x0_3)

    x0_1 = tf.compat.v1.image.resize_bilinear(x0_1, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0_2 = tf.compat.v1.image.resize_bilinear(x0_2, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0_3 = tf.compat.v1.image.resize_bilinear(x0_3, [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x0 = add([x[0], x0_1, x0_2, x0_3])
    x0 = Activation('relu')(x0)

    x1_0 = Conv2D(num_chan[1], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x1_0 = BatchNormalization()(x1_0)
    x1_2 = Conv2D(num_chan[1], 1, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x1_2 = BatchNormalization()(x1_2)
    x1_3 = Conv2D(num_chan[1], 1, padding='same', use_bias=False, kernel_initializer='he_normal')(x[3])

    x1_2 = tf.compat.v1.image.resize_bilinear(x1_2, [tf.shape(x[1])[1], tf.shape(x[1])[2]], align_corners=True)
    x1_3 = tf.compat.v1.image.resize_bilinear(x1_3, [tf.shape(x[1])[1], tf.shape(x[1])[2]], align_corners=True)
    x1 = add([x1_0, x[1], x1_2, x1_3])
    x1 = Activation('relu')(x1)

    x2_0 = Conv2D(num_chan[0], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x2_0 = BatchNormalization()(x2_0)
    x2_0 = Conv2D(num_chan[2], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x2_0)
    x2_0 = BatchNormalization()(x2_0)
    x2_1 = Conv2D(num_chan[2], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x2_1 = BatchNormalization()(x2_1)
    x2_3 = Conv2D(num_chan[2], 1, padding='same', use_bias=False, kernel_initializer='he_normal')(x[3])

    x2_3 = tf.compat.v1.image.resize_bilinear(x2_3, [tf.shape(x[2])[1], tf.shape(x[2])[2]], align_corners=True)
    x2 = add([x2_0, x2_1, x[2], x2_3])
    x2 = Activation('relu')(x2)

    x3_0 = Conv2D(num_chan[0], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[0])
    x3_0 = BatchNormalization()(x3_0)
    x3_0 = Conv2D(num_chan[0], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x3_0)
    x3_0 = BatchNormalization()(x3_0)
    x3_0 = Conv2D(num_chan[3], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x3_0)
    x3_0 = BatchNormalization()(x3_0)
    x3_1 = Conv2D(num_chan[1], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[1])
    x3_1 = BatchNormalization()(x3_1)
    x3_1 = Conv2D(num_chan[3], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x3_1)
    x3_1 = BatchNormalization()(x3_1)
    x3_2 = Conv2D(num_chan[3], 3, 2, padding='same', use_bias=False, kernel_initializer='he_normal')(x[2])
    x3_2 = BatchNormalization()(x3_2)

    x3 = add([x3_0, x3_1, x3_2, x[3]])
    x3 = Activation('relu')(x3)

    return [x0, x1, x2, x3]


# 最后的输出层
def final_layer(x, classes, size, activation):
    x0 = x[0]
    x1 = tf.compat.v1.image.resize_bilinear(x[1], [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x2 = tf.compat.v1.image.resize_bilinear(x[2], [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)
    x3 = tf.compat.v1.image.resize_bilinear(x[3], [tf.shape(x[0])[1], tf.shape(x[0])[2]], align_corners=True)

    x = concatenate([x0, x1, x2, x3], axis=-1)

    # x = Conv2D(x.shape[3], 3, 1, use_bias=False, padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = tf.compat.v1.image.resize_bilinear(x, size, align_corners=True)
    x = Conv2D(x.shape[3], 1, 1, use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(classes, 1, kernel_initializer='he_normal')(x)

    if activation in {'softmax', 'sigmoid'}:
        x = Activation(activation, name=activation)(x)

    return x


def cls_hrnet(batch_size,
              height,
              width,
              channel,
              classes,
              activation='softmax',
              hrnet_type='hrnet_w48'):
    if hrnet_type == 'hrnet_w18':
        size = [18, 36, 72, 144]
    elif hrnet_type == 'hrnet_w32':
        size = [32, 64, 128, 256]
    elif hrnet_type == 'hrnet_w48':
        size = [48, 96, 192, 384]
    else:
        raise ValueError("Unsupported hrnet type!")
    inputs = Input(batch_shape=(batch_size,) + (height, width, channel))

    x = stem_net(inputs)

    x = transition_layer1(x, size[:2])
    for i in range(1):
        x = make_branch1(x, size[:2])
        x = fuse_layer1(x, size[:2])

    x = transition_layer2(x, size[:3])
    for i in range(4):
        x = make_branch2(x, size[:3])
        x = fuse_layer2(x, size[:3])

    x = transition_layer3(x, size)
    for i in range(3):
        x = make_branch3(x, size)
        x = fuse_layer3(x, size)

    out = final_layer(x, classes=classes, size=(tf.shape(inputs)[1], tf.shape(inputs)[2]), activation=activation)

    model = Model(inputs=inputs, outputs=out)

    return model


if __name__ == "__main__":
    from tensorflow.keras.utils import plot_model
    import os
    os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz 2.44.1/bin/'

    model1 = cls_hrnet(batch_size=2, height=512, width=512, channel=3, classes=19, hrnet_type='hrnet_w48')
    model1.summary()
    plot_model(model1, to_file='./seg_hrnet.png', show_shapes=True)