import tensorflow as tf


# 交叉熵损失函数
def CE_Loss():
    def ce_loss(y_true, y_pred):
        # labels = tf.cast(tf.squeeze(labels), dtype=tf.int32)
        # cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
        loss = tf.reduce_mean(
            tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        return loss
    return ce_loss


# 带权重的交叉熵损失函数
def W_CELoss(weights=None):
    def w_celoss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)
        if weights is None:
            cross_entropy = tf.reduce_mean(
                tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        elif isinstance(weights, (float, int)):
            cross_entropy = weights * tf.reduce_mean(
                tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred))

        elif isinstance(weights, list):
            cross_entropy = tf.reduce_mean(
                weights * tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred))
        else:
            raise TypeError('unsupported weights')

        return cross_entropy
    return w_celoss


# Focal Loss
def FocalLoss(num_classes, alpha=0.25, gamma=2):
    def focalloss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)

        pos_loss = -alpha * tf.pow(1. - y_pred, gamma) * tf.math.log(tf.clip_by_value(y_pred, 1e-8, 1.0)) * y_true
        neg_loss = -(1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(tf.clip_by_value(1. - y_pred, 1e-8, 1.0)) * (
                1. - y_true)
        loss = tf.reduce_mean(pos_loss + neg_loss)
        return loss
    return focalloss


# OHEN交叉熵损失函数
def OHEM_CE_Loss(class_nums, top_k):
    """

    :param class_nums:
    :param thresh: 损失函数阈值, 0.65
    :param n_min: MIN_SAMPLE_NUMS: 262144    1:8
    :return:
    """
    def ohemce_loss(y_true, y_pred):
        y_true = tf.cast(tf.squeeze(y_true), dtype=tf.float32)

        # 计算cross entropy loss
        loss = tf.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
        loss, _ = tf.nn.top_k(loss, tf.size(loss), sorted=True)

        # apply ohem
        # ohem_thresh = tf.multiply(-1.0, tf.math.log(thresh))
        # ohem_cond = tf.greater(loss[n_min], ohem_thresh)
        # loss_select = tf.cond(pred=ohem_cond,
        #                       true_fn=lambda: tf.gather(loss, tf.squeeze(tf.where(tf.greater(loss, ohem_thresh)), 1)),
        #                       false_fn=lambda: loss[:n_min])
        loss_select = loss[:top_k]
        loss_value = tf.reduce_mean(loss_select)

        return loss_value
    return ohemce_loss


def get_loss(loss_type, classes, weights, top_k, alpha=0.25, gamma=2):
    if loss_type == 'ce_loss':
        loss = CE_Loss()
    elif loss_type == 'wce_loss':
        loss = W_CELoss(weights)
    elif loss_type == 'focal_loss':
        loss = FocalLoss(classes, alpha, gamma)
    elif loss_type == 'ohem_loss':
        loss = OHEM_CE_Loss(classes, top_k)
    else:
        raise TypeError('Unsupported loss type')

    return loss


