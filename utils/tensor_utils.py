"""
Utility functions for tensor related operations
"""

import logging
import os

import numpy as np
import scipy.io
import tensorflow as tf


def convert_label2onehot(gt, num_classes):
    """
    Util function used to convert labels in the one-hot format
    :param gt: 4D tensor: [batch_size, image_width, image_height, 1]
    :return: 4D tensor: [batch_size, image_width, image_height, num_classes]
    """
    # Prep the data. Make sure the labels are in one-hot format
    gt_one_hot = tf.squeeze(gt, axis=3)
    gt_one_hot = tf.cast(gt_one_hot, tf.uint8)
    gt_one_hot = tf.one_hot(gt_one_hot, num_classes)
    return gt_one_hot


def convert_val2onehot(gt, num_classes):
    """
    Util function used to convert labels in the one-hot format
    :param gt: 3D tensor: [image_width, image_height, 1]
    :return: 3D tensor: [image_width, image_height, num_classes]
    """
    # Prep the data. Make sure the labels are in one-hot format
    gt_one_hot = tf.squeeze(gt, axis=2)
    gt_one_hot = tf.cast(gt_one_hot, tf.uint8)
    gt_one_hot = tf.one_hot(gt_one_hot, num_classes)
    return gt_one_hot


def compute_and_print_IoU_per_class(confusion_matrix, num_classes, class_mask=None):
    """
    Computes and prints mean intersection over union divided per class
    :param confusion_matrix: confusion matrix needed for the computation
    """
    logging.basicConfig(level=logging.INFO)
    mIoU = 0
    out = ''
    index = ''
    true_classes = 0
    if class_mask == None:
        class_mask = np.ones([num_classes], np.int8)
    for i in range(num_classes):
        IoU = 0
        if class_mask[i] == 1:
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP

            denominator = (TP + FP + FN)
            # If the denominator is 0, we need to ignore the class.
            if denominator == 0:
                denominator = 1
            else:
                true_classes += 1
            IoU = TP / denominator
            mIoU += IoU
        index += '%7d' % i
        out += '%6.2f%%' % (IoU * 100)
    mIoU = mIoU / true_classes
    logging.info(' index :     ' + index)
    logging.info(' class IoU : ' + out)
    logging.info(' mIoU : %.2f%%' % (mIoU * 100))

    return mIoU


def save_matlab_files(step, npy_data):
    """
    Save numpy data to a Matlab format
    :param step: step of the data
    :param npy_data: data to save
    """
    npy_data = npy_data[0, :, :, :]
    try:
        os.makedirs('mat_files/')
    except os.error:
        pass
    scipy.io.savemat('mat_files/' + str(step) + '_G_softmax_output.mat', dict(x=npy_data))


def differentiable_argmax(logits):
    """
    Trick to obtain a differentiable argmax using softmax.

    :param logits: unprocessed tensor from the generator. 4D tensor: [batch_size, image_width, image_height, 3]
    :return: differentiable argmax of the imput logits. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    with tf.variable_scope('differentiable_argmax'):
        y = tf.nn.softmax(logits)
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, 3), k), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
        return y
