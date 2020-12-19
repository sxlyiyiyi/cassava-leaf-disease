import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from utils.data_aug import get_dataset
from utils.util import config_optimizer, learning_rate_config, add_regularization
from utils.util import ckpt_manager, set_device
from utils.logger import create_logger
from utils.score import ClassMetric
from tqdm import tqdm
from models.model_sum import get_model
from utils.loss import get_loss
from utils.visualization import show_image
from models.model_sum import get_model
import numpy as np
import os
import pathlib
from config import cfg


def train():
    set_device()
    output_dir = pathlib.Path(cfg.LOG_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)
    logger = create_logger(name=__name__,
                           output_dir=output_dir,
                           filename='log.txt')

    # 数据集加载
    train_dataset = get_dataset(cfg.DATASET.TRAIN_DATA, cfg, is_training=True)
    val_dataset = get_dataset(cfg.DATASET.VAL_DATA, cfg)

    # for batch, (images, labels) in enumerate(train_dataset):
    #     for i in range(cfg.TRAIN.BATCH_SIZE):
    #         show_image(images[i], labels[i], cfg.DATASET.LABELS)
    # endregion

    # 构建模型与损失函数
    model = get_model(cfg.MODEL_NAME, cfg)
    model = add_regularization(model, tf.keras.regularizers.l2(cfg.LOSS.WEIGHT_DECAY))
    model.summary()

    loss = get_loss(cfg.LOSS.TYPE, cfg)

    # 优化器和学习率配置
    lr = tf.Variable(cfg.SCHEDULER.LR_INIT)
    learning_rate = learning_rate_config(cfg)

    # warmup策略
    def lr_with_warmup(global_steps):
        lr_ = tf.cond(tf.less(global_steps, cfg.SCHEDULER.WARMUP_STEPS),
                      lambda: cfg.SCHEDULER.LR_INIT * tf.cast((global_steps + 1) / cfg.SCHEDULER.WARMUP_STEPS,
                                                              tf.float32),
                      lambda: tf.maximum(learning_rate(global_steps - cfg.SCHEDULER.WARMUP_STEPS),
                                         cfg.SCHEDULER.LR_LOWER_BOUND))
        return lr_

    optimizer = config_optimizer(cfg, learning_rate=lr)

    # 模型保存与恢复
    manager, ckpt = ckpt_manager(cfg, model, logger, optimizer)

    # endregion

    # region # 训练与验证静态图
    @tf.function
    def train_one_batch(x, y):
        with tf.GradientTape() as tape:
            # 1、计算模型输出和损失
            pred_o = model(x, training=True)
            regularization_loss_out = tf.reduce_sum(model.losses)
            train_loss_out = loss(y, pred_o)
            total_loss_out = train_loss_out + regularization_loss_out
        # 计算梯度以及更新梯度, 固定用法
        grads = tape.gradient(total_loss_out, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss_out, train_loss_out, pred_o

    @tf.function
    def val_one_batch(x):
        pred_o = model(x, training=False)
        return pred_o
    # endregion

    # region # 记录器和评价指标
    summary_writer = tf.summary.create_file_writer(cfg.LOG_DIR)
    # tf.summary.trace_on(profiler=True)  # 开启Trace（可选）
    train_metric = ClassMetric(cfg.DATASET.N_CLASSES)
    val_metric = ClassMetric(cfg.DATASET.N_CLASSES)
    # endregion

    # region # 迭代优化
    for _ in range(int(ckpt.step), cfg.TRAIN.EPOCHS):
        # region  # 训练集
        ckpt.step.assign_add(1)
        lr.assign(lr_with_warmup(optimizer.iterations))  # 必须使用assign才能改变optimizer的lr的值，否则，是个固定值

        for batch, (images, labels) in tqdm(enumerate(train_dataset)):
            total_loss, train_loss, train_pred = train_one_batch(images, labels)
            if int(ckpt.step) % cfg.TRAIN.SNAP_SHOT == 1:
                train_out = np.argmax(train_pred, axis=-1)
                train_metric.addBatch(labels, train_out)
            with summary_writer.as_default():  # 指定记录器
                tf.summary.scalar("train/total_loss", total_loss, step=optimizer.iterations)  # 将当前损失函数的值写入记录器
                tf.summary.scalar("train/train_loss", train_loss, step=optimizer.iterations)
                tf.summary.scalar("train/learning_rate", lr, step=optimizer.iterations)
        # endregion

        # region # 验证集
        if int(ckpt.step) % cfg.TRAIN.SNAP_SHOT == 1:
            for batch, (images, labels) in tqdm(enumerate(val_dataset)):
                val_pred = val_one_batch(images)
                val_out = np.argmax(val_pred, axis=-1)
                val_metric.addBatch(labels, val_out)

            with summary_writer.as_default():
                # 保存Trace信息到文件（可选）
                # tf.summary.trace_export(name="model_trace", step=epoch, profiler_outdir=FLAGS.logs_dir)
                tf.summary.scalar("train_metrics/f1_score", train_metric.f1score(), step=int(ckpt.step))
                tf.summary.scalar("train_metrics/Acc", train_metric.acc(), step=int(ckpt.step))
                tf.summary.scalar("val_metrics/f1_score", val_metric.f1score(), step=int(ckpt.step))
                tf.summary.scalar("val_metrics/Acc", val_metric.acc(), step=int(ckpt.step))
                val_f1 = val_metric.f1score()
                logger.info("epoch: {}, train_f1:{:.5f}, val_f1:{:.5f}, train_acc:{:.5f}, val_acc:{:.5f}".format(
                    int(ckpt.step), train_metric.f1score(), val_metric.f1score(), train_metric.acc(), val_metric.acc()
                ))
            train_metric.reset()
            val_metric.reset()
        # endregion

        # region # 模型保存
        # 使用CheckpointManager保存模型参数到文件并自定义编号
        path = manager.save()

        # if val_f1 >= 0.98:
        #     os.mkdir(ckpt_dir + '//epoch_{}_train_{:.6f}'.format(epoch, val_f1))
        #     tf.keras.models.save_model(
        #         model, ckpt_dir + '//epoch_{}_train_{:.6f}'.format(epoch, val_f1))
        # endregion
    # endregion


if __name__ == '__main__':
    train()