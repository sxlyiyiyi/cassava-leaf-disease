import tensorflow_addons as tfa
import tensorflow as tf

def rot90(img, lab):
    i = tf.random.shuffle([0, 1, 2, 3])
    img = tf.image.rot90(img, i[0])

    return img, lab


def random_rotate(img, label, rotation):
    angle = tf.random.uniform([], minval=-rotation, maxval=rotation)
    img = tfa.image.rotate(img, angle, interpolation='BILINEAR')

    return img, label
