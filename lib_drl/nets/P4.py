# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
import numpy as np

from nets.network import Network
from model.config import cfg

pyramid_maps = {
    'resnet50': {'C1': 'resnet_v1_50/pool1/Relu:0',
                 'C2': 'resnet_v1_50/block1/unit_2/bottleneck_v1',
                 'C3': 'resnet_v1_50/block2/unit_3/bottleneck_v1',
                 'C4': 'resnet_v1_50/block3/unit_5/bottleneck_v1',
                 'C5': 'resnet_v1_50/block4/unit_3/bottleneck_v1',
                 },
    'resnet101': {'C1': '', 'C2': '',
                  'C3': '', 'C4': '',
                  'C5': '',
                  }
}


def resnet_arg_scope(is_training=True,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': tf.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            weights_initializer=slim.variance_scaling_initializer(),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


def fusion_two_layer(C_i, P_j, scope):
    '''
    i = j+1
    :param C_i: shape is [1, h, w, c]
    :param P_j: shape is [1, h/2, w/2, 256]
    :return:
    P_i
    '''
    with tf.variable_scope(scope):
        level_name = scope.split('_')[1]
        h, w = tf.shape(C_i)[1], tf.shape(C_i)[2]
        upsample_p = tf.image.resize_bilinear(P_j,
                                              size=[h, w],
                                              name='up_sample_' + level_name)

        reduce_dim_c = slim.conv2d(C_i,
                                   num_outputs=256,
                                   kernel_size=[1, 1], stride=1,
                                   scope='reduce_dim_' + level_name)

        add_f = 0.5 * upsample_p + 0.5 * reduce_dim_c

        # P_i = slim.conv2d(add_f,
        #                   num_outputs=256, kernel_size=[3, 3], stride=1,
        #                   padding='SAME',
        #                   scope='fusion_'+level_name)
        return add_f


class P4(Network):
    def __init__(self, num_layers=50):
        Network.__init__(self)
        self._feat_stride = cfg.ANCHOR_STRIDES
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._num_layers = num_layers
        self._scope = 'resnet_v1_%d' % num_layers
        self._decide_blocks()

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def _build_base(self):
        with tf.variable_scope(self._scope, self._scope):
            net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    def _image_to_head(self, is_training, reuse=None):
        assert (0 <= cfg.RESNET.FIXED_BLOCKS <= 3)
        # Now the base is always fixed during training
        if self._scope == 'resnet_v1_50':
            middle_num_units = 6
        elif self._scope == 'resnet_v1_101':
            middle_num_units = 23
        else:
            raise NotImplementedError('We only support resnet_v1_50 or resnet_v1_101. Check your network name....yjr')
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net_conv = self._build_base()
        # if cfg.RESNET.FIXED_BLOCKS > 0:
        #     with slim.arg_scope(resnet_arg_scope(is_training=False)):
        #         net_conv, _ = resnet_v1.resnet_v1(net_conv,
        #                                           self._blocks[0:cfg.RESNET.FIXED_BLOCKS],
        #                                           global_pool=False,
        #                                           include_root_block=False,
        #                                           reuse=reuse,
        #                                           scope=self._scope)
        # if cfg.RESNET.FIXED_BLOCKS < 3:
        #     with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
        #         net_conv, _ = resnet_v1.resnet_v1(net_conv,
        #                                           self._blocks[cfg.RESNET.FIXED_BLOCKS:-1],
        #                                           global_pool=False,
        #                                           include_root_block=False,
        #                                           reuse=reuse,
        #                                           scope=self._scope)

        not_freezed = [False] * cfg.RESNET.FIXED_BLOCKS + (4 - cfg.RESNET.FIXED_BLOCKS) * [True]
        # Fixed_Blocks can be 1~3
        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[0]))):
            C2, end_points_C2 = resnet_v1.resnet_v1(net_conv,
                                                    self._blocks[0:1],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._scope)

        # C2 = tf.Print(C2, [tf.shape(C2)], summarize=10, message='C2_shape')

        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[1]))):
            C3, end_points_C3 = resnet_v1.resnet_v1(C2,
                                                    self._blocks[1:2],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._scope)

        # C3 = tf.Print(C3, [tf.shape(C3)], summarize=10, message='C3_shape')
        with slim.arg_scope(resnet_arg_scope(is_training=(is_training and not_freezed[2]))):
            C4, end_points_C4 = resnet_v1.resnet_v1(C3,
                                                    self._blocks[2:3],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._scope)

        # C4 = tf.Print(C4, [tf.shape(C4)], summarize=10, message='C4_shape')
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            C5, end_points_C5 = resnet_v1.resnet_v1(C4,
                                                    self._blocks[3:4],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._scope)
        # C5 = tf.Print(C5, [tf.shape(C5)], summarize=10, message='C5_shape')

        feature_dict = {'C2': end_points_C2['{}/block1/unit_2/bottleneck_v1'.format(self._scope)],
                        'C3': end_points_C3['{}/block2/unit_3/bottleneck_v1'.format(self._scope)],
                        'C4': end_points_C4[
                            '{}/block3/unit_{}/bottleneck_v1'.format(self._scope, middle_num_units - 1)],
                        'C5': end_points_C5['{}/block4/unit_3/bottleneck_v1'.format(self._scope)],
                        # 'C5': end_points_C5['{}/block4'.format(scope_name)],
                        }

        # build pyramid
        pyramid_dict = {}
        with tf.variable_scope('build_pyramid'):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                P5 = slim.conv2d(C5,
                                 num_outputs=256,
                                 kernel_size=[1, 1],
                                 stride=1, scope='build_P5')
                pyramid_dict['P5'] = P5
                pyramid_dict['P4'] = fusion_two_layer(C_i=feature_dict["C4"],
                                                      P_j=pyramid_dict["P5"],
                                                      scope='build_P4')
                pyramid_dict['P4'] = slim.conv2d(pyramid_dict['P4'],
                                                 num_outputs=256, kernel_size=[3, 3], padding="SAME",
                                                 stride=1, scope="fuse_P4")

        # return [P4, P5]
        print("we are in Pyramid::-======>>>>")
        print(pyramid_dict.keys())
        print("base_anchor_size are: ", cfg.ANCHOR_SIZES)
        print(20 * "__")
        # self._act_summaries.append([pyramid_dict[level_name] for level_name in cfg.FPN.LEVELS])
        self._layers['head'] = pyramid_dict['P4']
        # return pyramid_dict  # return the dict. And get each level by key. But ensure the levels are consitant
        # return list rather than dict, to avoid dict is unordered
        return pyramid_dict['P4']

    def _head_to_tail(self, pool5, is_training, reuse=None):
        with tf.variable_scope('build_fc_layers'):
            # fc1 = slim.conv2d(inputs=inputs,
            #                   num_outputs=1024,
            #                   kernel_size=[7, 7],
            #                   padding='VALID',
            #                   scope='fc1') # shape is [minibatch_size, 1, 1, 1024]
            # fc1 = tf.squeeze(fc1, [1, 2], name='squeeze_fc1')

            inputs = slim.flatten(inputs=pool5, scope='flatten_inputs')

            fc6 = slim.fully_connected(inputs, num_outputs=1024, scope='fc6')

            fc7 = slim.fully_connected(fc6, num_outputs=1024, scope='fc7')

            # fc3 = slim.fully_connected(fc2, num_outputs=1024, scope='fc3')

            # we add fc3 to increase the ability of fast-rcnn head
            return fc7

    def _decide_blocks(self):
        # choose different blocks for different number of layers
        if self._num_layers == 50:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 101:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=23, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        elif self._num_layers == 152:
            self._blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                            resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
                            # use stride 1 for the last conv4 layer
                            resnet_v1_block('block3', base_depth=256, num_units=36, stride=1),
                            resnet_v1_block('block4', base_depth=512, num_units=3, stride=1)]

        else:
            # other numbers are not supported
            raise NotImplementedError

    def get_variables_to_restore(self, variables, var_keep_dic):
        variables_to_restore = []

        for v in variables:
            if v.name.split(':')[0] in var_keep_dic:
                print('Variables restored: %s' % v.name)
                variables_to_restore.append(v)

        return variables_to_restore

