# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
import numpy as np

from nets.network import Network
from model.config import cfg


class vgg16(Network):
    def __init__(self):
        Network.__init__(self)
        self._scope = 'vgg_16'

    def _image_to_head(self, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            net = slim.repeat(self._image, 2, slim.conv2d, 64, [3, 3],
                              trainable=False, scope='conv1')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3],
                              trainable=False, scope='conv2')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3],
                              trainable=is_training, scope='conv3')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv4')
            net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3],
                              trainable=is_training, scope='conv5')
        self._layers['head'] = net
        return net

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope(self._scope, self._scope, reuse=reuse):
            pool5_flat = slim.flatten(pool5, scope='flatten')
            fc6 = slim.fully_connected(pool5_flat, 4096, scope='fc6')
            if is_training:
                fc6 = slim.dropout(fc6, keep_prob=0.5, is_training=True,
                                   scope='dropout6')
            fc7 = slim.fully_connected(fc6, 4096, scope='fc7')
            if is_training:
                fc7 = slim.dropout(fc7, keep_prob=0.5, is_training=True,
                                   scope='dropout7')
        return fc7

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)
        return variables_to_restore
