# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Partially written* by Chang Hsiao-Chien
# Partially written* by  Aleksis Pirinen, rest is based on original code by
# Xinlei Chen.
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

import numpy as np
from time import sleep

from layer_utils.roi_align import roi_align
from layer_utils.snippets import generate_anchors_fixate
from layer_utils.proposal_layer import proposal_layer_fixate
from layer_utils.proposal_target_layer import proposal_target_layer_wo_scores
from model.config import cfg
from model.reward_functions import reward_fixate, reward_done
from model.factory import sample_fix_loc


class Network(object):
    def __init__(self):
        self._predictions = {}
        self._losses = {}
        self._anchor_targets = {}
        self._proposal_targets = {}
        self._layers = {}
        self._gt_image = None
        self._variables_to_fix = {}

    def _add_gt_image(self):
        # add back mean
        image = self._image + cfg.PIXEL_MEANS
        # BGR to RGB (opencv uses BGR)
        resized = tf.image.resize_bilinear(image, tf.to_int32(self._im_info[:2] \
                                                              / self._im_info[2]))
        self._gt_image = tf.reverse(resized, axis=[-1])

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:
            # change the channel to the caffe format
            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])
            # then force it to have channel 2
            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1],
                                                            [input_shape[2]]]))
            # then swap the channel back
            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)

    def _proposal_layer_fixate(self, anchors, rpn_bbox_pred, name='proposal_all'):
        with tf.variable_scope(name) as scope:
            rois_fixate, _, _ = tf.py_func(proposal_layer_fixate,
                                           [rpn_bbox_pred, self._im_info,
                                            anchors],
                                           [tf.float32, tf.int32, tf.int32],
                                           name=name)
        return rois_fixate

    def _crop_pool_layer(self, bottom, rois, _im_info, name):
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            img_h, img_w = tf.cast(_im_info[0], tf.float32), tf.cast(_im_info[1], tf.float32)
            # N = tf.shape(rois)[0]
            _, x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)
            pre_pool_size = cfg.POOLING_SIZE * 2
            cropped_roi_features = tf.image.crop_and_resize(bottom, normalized_rois,
                                                            box_ind=tf.to_int32(batch_ids),
                                                            crop_size=[pre_pool_size, pre_pool_size],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features, [2, 2], stride=2, padding='SAME')
        return roi_features

    def _proposal_target_layer_wo_scores(self, rois, name):
        with tf.variable_scope(name) as scope:
            rois, labels, bbox_targets, bbox_inside_weights, \
            bbox_outside_weights = tf.py_func(proposal_target_layer_wo_scores,
                                              [rois, self._gt_boxes, self._num_classes],
                                              [tf.float32, tf.float32, tf.float32,
                                               tf.float32, tf.float32],
                                              name="proposal_target_wo")

            rois.set_shape([cfg.TRAIN.BATCH_SIZE, 5])
            labels.set_shape([cfg.TRAIN.BATCH_SIZE, 1])
            bbox_targets.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_inside_weights.set_shape([cfg.TRAIN.BATCH_SIZE, self._num_classes * 4])
            bbox_outside_weights.set_shape([cfg.TRAIN.BATCH_SIZE,
                                            self._num_classes * 4])

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = tf.to_int32(labels, name="to_int32")
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights
            return rois

    def _anchor_component_fixate(self, height, width, h_start, w_start):
        with tf.variable_scope('ANCHOR_' + self._tag) as scope:
            # just to get the shape right
            anchors, anchor_length = tf.py_func(generate_anchors_fixate,
                                                [height, width, h_start, w_start,
                                                 self._anchor_strides[0],
                                                 self._anchor_sizes,
                                                 self._anchor_ratios],
                                                [tf.float32, tf.int32], name="generate_anchors")
            anchors.set_shape([None, 4])
            anchor_length.set_shape([])
            self._anchors = anchors
            self._anchor_length = anchor_length

    def _smooth_l1_loss(self, bbox_pred, bbox_targets, bbox_inside_weights,
                        bbox_outside_weights, sigma=1.0, dim=[1]):
        sigma_2 = sigma ** 2
        box_diff = bbox_pred - bbox_targets
        in_box_diff = bbox_inside_weights * box_diff
        abs_in_box_diff = tf.abs(in_box_diff)
        smoothL1_sign = tf.stop_gradient(tf.to_float(tf.less(abs_in_box_diff, 1. \
                                                             / sigma_2)))
        in_loss_box = tf.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                      + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
        out_loss_box = bbox_outside_weights * in_loss_box
        loss_box = tf.reduce_mean(tf.reduce_sum(out_loss_box, axis=dim))
        return loss_box

    # Currently only supports training of detector head
    def _add_losses(self):
        with tf.variable_scope('LOSS_' + self._tag) as scope:
            # RCNN, class loss
            cls_score = self._predictions['cls_score_seq']
            label = tf.reshape(self._proposal_targets['labels'], [-1])
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=cls_score, labels=label))
            # RCNN, bbox loss
            bbox_pred = self._predictions['bbox_pred_seq']
            bbox_targets = self._proposal_targets['bbox_targets']
            bbox_inside_weights = self._proposal_targets['bbox_inside_weights']
            bbox_outside_weights = self._proposal_targets['bbox_outside_weights']
            loss_box = self._smooth_l1_loss(bbox_pred, bbox_targets,
                                            bbox_inside_weights,
                                            bbox_outside_weights)
            self._losses['cross_entropy'] = cross_entropy
            self._losses['loss_box'] = loss_box
            loss = cross_entropy + loss_box
            # Want to use regularizer, but only for the actual weights used in the
            # detector updates!
            all_reg_losses = tf.losses.get_regularization_losses()
            print(self._scope)
            ii = 0
            if cfg.P4 or self._scope == 'vgg_16':
                while 'fc6' not in all_reg_losses[ii].name:
                    ii += 1
            else:
                while 'block4' not in all_reg_losses[ii].name:
                    ii += 1
            relevant_reg_losses = all_reg_losses[ii:]
            reg_loss = tf.add_n(relevant_reg_losses, 'regu')
            self._losses['total_loss'] = loss + reg_loss

    def _region_proposal_fixate(self, net_conv, is_training, initializer):
        rpn = slim.conv2d(net_conv, cfg.RPN_CHANNELS, [3, 3], trainable=is_training,
                          weights_initializer=initializer, scope='rpn_conv/3x3')
        rpn_bbox_pred = slim.conv2d(rpn, self._num_anchors * 4, [1, 1],
                                    trainable=is_training,
                                    weights_initializer=initializer,
                                    padding='VALID', activation_fn=None,
                                    scope='rpn_bbox_pred')
        self._predictions["rpn_bbox_pred"] = rpn_bbox_pred

    def _region_classification(self, fc7, is_training, initializer,
                               initializer_bbox, reuse=None):
        cls_score = slim.fully_connected(fc7, self._num_classes,
                                         weights_initializer=initializer,
                                         trainable=is_training, reuse=reuse,
                                         activation_fn=None, scope='cls_score')
        cls_prob = self._softmax_layer(cls_score, "cls_prob")
        cls_pred = tf.argmax(cls_score, axis=1, name="cls_pred")
        bbox_pred = slim.fully_connected(fc7, self._num_classes * 4,
                                         weights_initializer=initializer_bbox,
                                         trainable=is_training, reuse=reuse,
                                         activation_fn=None, scope='bbox_pred')
        self._predictions['cls_score_seq'] = cls_score
        self._predictions['cls_pred_seq'] = cls_pred
        self._predictions['cls_prob_seq'] = cls_prob
        self._predictions['bbox_pred_seq'] = bbox_pred

    ############# LRP-HAI ADDITIONAL COMPONENTS -- START #########################

    def train_LRP_HAI(self, sess, lr_rl, sc, stats):
        # Compute baseline
        if cfg.LRP_HAI_TRAIN.USE_BL:
            # cfg.LRP_HAI.MAX_ITER_TRAJ = 13
            bl_means_total = np.empty(cfg.LRP_HAI.MAX_ITER_TRAJ)
            bl_stds_total = np.empty(cfg.LRP_HAI.MAX_ITER_TRAJ)
            for idx in range(cfg.LRP_HAI.MAX_ITER_TRAJ):
                if len(self._bl_total[idx]) > 0:
                    bl_means_total[idx] = np.mean(self._bl_total[idx])
                    bl_stds_total[idx] = np.std(self._bl_total[idx])
            bl_stds_total[bl_stds_total == 0] = 1

        curr_batch_avg_loss = 0
        curr_batch_ce_done_rew_prod = 0
        curr_batch_ce_fix_rew_prod = 0

        for idx in range(len(self._ep_batch['x'])):
            # Potentially normalize to mean 0, std 1
            ep_rew_total = self._ep_batch['total_return'][idx]

            # rew = (rew - bl_mean) / std
            if cfg.LRP_HAI_TRAIN.USE_BL:
                ep_rew_total -= bl_means_total[: len(ep_rew_total)]
                ep_rew_total /= bl_stds_total[: len(ep_rew_total)]

            # Update grad buffer
            feed_dict_grad_comp \
                = {self._rl_in: self._ep_batch['x'][idx],
                   self._rl_hid: self._ep_batch['h'][idx],
                   self._done_labels: self._ep_batch['done'][idx],
                   self._fix_labels: self._ep_batch['fix'][idx],
                   self._advs_total: self._ep_batch['total_return'][idx],
                   self._cond_switch_fix: self._ep_batch['cond'][idx]}

            ce_done, ce_fix, ce_done_rew_prod, ce_fix_rew_prod, loss_rl, new_grads \
                = sess.run([self._predictions['ce_done'],
                            self._predictions['ce_fix'],
                            self._predictions['ce_done_rew_prod'],
                            self._predictions['ce_fix_rew_prod'],
                            self._predictions['loss_rl'],
                            self._predictions['new_grads']],
                           feed_dict=feed_dict_grad_comp)
            curr_batch_avg_loss += loss_rl
            curr_batch_ce_done_rew_prod += ce_done_rew_prod
            curr_batch_ce_fix_rew_prod += ce_fix_rew_prod

            # Accumulate gradients to buffer
            for ix, grad in enumerate(new_grads):
                self._grad_buffer[ix] += grad

        curr_batch_avg_loss /= cfg.LRP_HAI_TRAIN.BATCH_SIZE
        curr_batch_ce_done_rew_prod /= cfg.LRP_HAI_TRAIN.BATCH_SIZE
        curr_batch_ce_fix_rew_prod /= cfg.LRP_HAI_TRAIN.BATCH_SIZE

        sc.update(curr_batch_avg_loss, curr_batch_ce_done_rew_prod, curr_batch_ce_fix_rew_prod, stats)

        # Update policy parameters
        feed_dict_upd_grads \
            = {self._batch_grad[grad_idx]: self._grad_buffer[grad_idx] \
               for grad_idx in range(len(self._batch_grad))}
        feed_dict_upd_grads.update({self._lr_rl_in: lr_rl})
        sess.run(self._update_grads, feed_dict=feed_dict_upd_grads)

        # Reset gradient buffer etc.
        self.reset_after_gradient()

    def _collect_traj(self, t, free_will, nbr_gts):
        # Stack inputs, hidden states, action grads, and rewards for this episode
        epx = np.vstack(self._ep['x'])
        eph = np.vstack(self._ep['h'])
        ep_done = np.vstack([0] * t + [free_will])
        ep_fix = np.hstack(self._ep['fix'])
        ep_rew_done = np.hstack(self._ep['rew_done'])
        ep_rew_fix = np.hstack(self._ep['rew_fix'])
        ep_total_reward = ep_rew_fix + ep_rew_done
        # cumulative reward
        discounted_ep_total_reward = np.zeros_like(ep_total_reward)
        running_add = 0
        for t in reversed(range(0, len(ep_total_reward))):
            running_add = running_add * cfg.LRP_HAI_TRAIN.P_GAMMA + ep_total_reward[t]
            discounted_ep_total_reward[t] = running_add

        # This ensures that images with many gt's are not "more valuable" than
        # images with few gts
        if nbr_gts > 0:
            discounted_ep_total_reward /= nbr_gts

        # Add to collections
        self._ep_batch['x'].append(epx)
        self._ep_batch['h'].append(eph)
        self._ep_batch['done'].append(ep_done)
        self._ep_batch['fix'].append(ep_fix)
        self._ep_batch['total_return'].append(discounted_ep_total_reward)
        self._ep_batch['cond'].append(int(free_will))

        # Add to baselines
        if cfg.LRP_HAI_TRAIN.USE_BL:
            for len_ctr in range(len(ep_total_reward)):
                self._bl_total[len_ctr].append(discounted_ep_total_reward[len_ctr])

    # Done reward
    def reward_done(self, fix_prob, t, gt_max_ious, free_will=True):
        if free_will:
            # The fixate labels need to be handled with care,
            # depending on whether or not we terminated by an
            # action (done-action) or by running out of iterations!
            self._ep['fix'].append(0)
            rew_done = reward_done(gt_max_ious)
            rew_fixate = 0.0
            self._ep['rew_done'].append(rew_done)
            # Fixate rewards should never be held "accountable"
            # due to stopping condition
            self._ep['rew_fix'].append(rew_fixate)
        else:
            _, _, fix_one_hot = sample_fix_loc(fix_prob, 'train')
            self._ep['fix'].append(fix_one_hot)
            rew_fixate = 0.0
            rew_done = -0.5  # so agent may learn stop before forced
            self._ep['rew_fix'].append(rew_fixate)
            self._ep['rew_done'].append(rew_done)

        # Now we also collect all throughout this trajectory
        self._collect_traj(t, free_will, gt_max_ious.shape[0])

        # Return reward
        return rew_done, rew_fixate

    # Fixate reward
    def reward_fixate(self, pred_bboxes, gt_boxes, gt_max_ious, t, beta):

        # Separation of rewards (described in paper) does not appear useful when
        # training for various beta exploration penalties, so not used here
        rew_done = -beta
        self._ep['rew_done'].append(rew_done)

        # Fixate reward
        rew_fixate, gt_max_ious = reward_fixate(pred_bboxes, gt_boxes, gt_max_ious)
        self._ep['rew_fix'].append(rew_fixate)

        return rew_fixate, rew_done, gt_max_ious

    # Below are two helper function used, depending on whether agent
    # terminates by done action or by exceeding max-length trajectory
    def ce_fix_terminate_via_max_it(self, fix_logits):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=fix_logits, labels=self._fix_labels, name="ce_fix1")

    # In this case, fix_labels' final entry is wrong and we need to get
    # rid of that part (setting cross-entropy manually to zero)
    def ce_fix_terminate_via_done(self, fix_logits):
        ce_fix = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=fix_logits, labels=self._fix_labels, name="ce_fix2")
        return tf.concat([tf.slice(ce_fix, [0], [tf.shape(ce_fix)[0] - 1]),
                          tf.zeros([1])], 0)

    def reset_after_gradient(self):
        for ix, grad in enumerate(self._grad_buffer):
            self._grad_buffer[ix] = grad * 0
        self._ep_batch = {'x': [], 'h': [], 'done': [], 'fix': [], 'cond': [], 'total_return': []}
        self._bl_total = [[] for _ in range(cfg.LRP_HAI.MAX_ITER_TRAJ)]

    def reset_pre_traj(self):
        # xs = collected observations, hs, is corresponding hidden,
        # ys = collected "fake labels", rews = collected rewards
        self._ep = {'x': [], 'h': [], 'fix': [], 'rew_done': [],
                    'rew_fix': [], 'rew_done_gt': [], 'rew_fix_gt': []}

    def _compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var)
                for var, grad in zip(var_list, grads)]

    def init_rl_train(self, sess):
        # Return RL-trainable variables (thus skip detector parameters here;
        # they are treated separately).
        self.drl_tvars = tf.trainable_variables()
        for i in self.fr_tvars:
            self.drl_tvars.remove(i)
        print("LRP-HAI trainable variables")
        for i in self.drl_tvars:
            print(i.name)

        self._batch_grad = [tf.placeholder(tf.float32,
                                           name='LRP_HAI_grad_' + str(idx)) \
                            for idx in range(len(self.drl_tvars))]
        # Optimizer
        temp = set(tf.global_variables())
        self._lr_rl_in = tf.placeholder(tf.float32)
        adam = tf.train.AdamOptimizer(learning_rate=cfg.LRP_HAI_TRAIN.LEARNING_RATE)
        self._update_grads = adam.apply_gradients(zip(self._batch_grad, self.drl_tvars))
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp),
                 feed_dict={self._lr_rl_in: cfg.LRP_HAI_TRAIN.LEARNING_RATE})

        # RL loss
        self._done_labels = tf.placeholder(tf.float32, [None, 1], name="done_labels")
        self._fix_labels = tf.placeholder(tf.int32, [None], name="fix_labels")
        self._advs_total = tf.placeholder(tf.float32, [None], name="total_return")
        self._cond_switch_fix = tf.placeholder(tf.int32)

        done_logits = self._predictions['done_logits']
        fix_logits = self._predictions['fix_logits']

        ce_done = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self._done_labels, logits=done_logits,
            name="ce_done_logits"))
        ce_fix = tf.cond(tf.equal(self._cond_switch_fix, 0),
                         lambda: self.ce_fix_terminate_via_max_it(fix_logits),
                         lambda: self.ce_fix_terminate_via_done(fix_logits))
        ce_done_rew_prod = ce_done * self._advs_total
        ce_fix_rew_prod = ce_fix * self._advs_total

        ce_done_rew_prod = tf.reduce_sum(ce_done_rew_prod)
        ce_fix_rew_prod = tf.reduce_sum(ce_fix_rew_prod)

        ce_rew_prod = ce_done_rew_prod + ce_fix_rew_prod
        loss_rl = tf.reduce_sum(ce_rew_prod)
        new_grads = self._compute_gradients(loss_rl, self.drl_tvars)

        # Add to predictions container
        self._predictions['ce_done'] = ce_done
        self._predictions['ce_fix'] = ce_fix
        self._predictions['ce_done_rew_prod'] = ce_done_rew_prod
        self._predictions['ce_fix_rew_prod'] = ce_fix_rew_prod
        self._predictions['loss_rl'] = loss_rl
        self._predictions['new_grads'] = new_grads

        # Initialize gradient buffer
        self._grad_buffer = sess.run(self.drl_tvars)

        # Reset the gradient placeholder and other parts
        self.reset_after_gradient()

    def _net_rois_batched(self):
        return self._rois_seq_batched

    def _net_rois_seq(self):
        return self._rois_seq

    def _build_network(self, is_training=True):
        # select initializers
        if cfg.TRAIN.TRUNCATED:
            initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.truncated_normal_initializer(mean=0.0, stddev=0.001)
        else:
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            initializer_bbox = tf.random_normal_initializer(mean=0.0, stddev=0.001)

        # Note: We create both net_conv given from input image (image_to_head)
        # and as a placeholder, since we also want to be able to immediately
        # send in the precomputed feature map at some stages
        net_conv = self._image_to_head(False)
        self._predictions['net_conv'] = net_conv
        self._net_conv_in \
            = tf.placeholder(tf.float32, shape=[None, None, None, cfg.DIMS_BASE])

        # Below conditional is used as follows: The first time an image is sent
        # through the network, net_conv above is produced. But if we later want
        # to use that conv-map again (e.g. when training detector in LRP-HAI
        # training), we don't want to have to send the full image again, and instead
        # directly use the pre-computed feature map.

        # TODO: Note that due to the tensorflow graph construction,
        # both branches will be executed despite using cond -- potentially could
        # gain speedup by having separate graph construction during inferences

        # Similar story for the below conditional
        self._cond_switch_roi = tf.placeholder(tf.int32)
        self._rois_seq = tf.placeholder(tf.float32, shape=[None, 5])
        self._rois_seq_batched \
            = self._proposal_target_layer_wo_scores(self._rois_seq, 'rois_seq_batched')
        rois_in = tf.cond(tf.equal(self._cond_switch_roi, 0),
                          lambda: self._net_rois_batched(),
                          lambda: self._net_rois_seq())

        # Sequential class-specific processing
        if cfg.POOLING_MODE == 'roi_pooling':
            pool5_LRP_HAI = self._crop_pool_layer(self._net_conv_in, rois_in, self._im_info, "pool5_LRP_HAI")
        elif cfg.POOLING_MODE == 'roi_align':
            pool5_LRP_HAI = roi_align(self._net_conv_in, rois_in, self._anchor_strides[0], cfg.POOLING_SIZE)
        else:
            raise NotImplementedError

        fc7_seq = self._head_to_tail(pool5_LRP_HAI, is_training)
        with tf.variable_scope(self._scope, self._scope):
            self._region_classification(fc7_seq, is_training, initializer,
                                        initializer_bbox)

    def build_LRP_HAI_network(self, is_training=True):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        # Initial processing
        net_conv = self._predictions['net_conv']
        # Convolutional GRU
        self.HAI(is_training, tf.contrib.layers.xavier_initializer())

        # build rpn for fixate region
        self.fix_rect_h = tf.placeholder(tf.int32)
        self.fix_rect_w = tf.placeholder(tf.int32)
        self.h_start = tf.placeholder(tf.int32)
        self.w_start = tf.placeholder(tf.int32)
        self._net_conv_fixate = tf.placeholder(tf.float32, shape=[None, None, None, cfg.DIMS_BASE])

        with tf.variable_scope(self._scope, self._scope):
            # build the anchors for the image
            self._anchor_component_fixate(self.fix_rect_h, self.fix_rect_w, self.h_start, self.w_start)
            # region proposal network
            self._region_proposal_fixate(self._net_conv_fixate, False, initializer)

        rpn_bbox_pred = self._predictions['rpn_bbox_pred']
        anchors = self._anchors
        self._initial_rl_input(net_conv, anchors, rpn_bbox_pred)

    # Shorter way of writing tf.get_variable(...)
    def _make_var(self, name, shape, initializer=None, is_training=True):
        return tf.get_variable(name, shape, dtype=None, initializer=initializer,
                               regularizer=None, trainable=is_training)

    def scaled_dot_product_attention(self, Q, K, V,
                                     scope="scaled_dot_product_attention"):
        """See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= (d_k ** 0.5)

            # softmax
            outputs = tf.nn.softmax(outputs)
            # visualize attention
            # attention = tf.transpose(outputs, [0, 2, 1])
            # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)
        return outputs

    def scaled_dot_product_attention_1(self, Q, K, V,
                                       scope="scaled_dot_product_attention_1"):
        """See 3.2.1.
        Q: Packed queries. 3d tensor. [N, T_q, d_k].
        K: Packed keys. 3d tensor. [N, T_k, d_k].
        V: Packed values. 3d tensor. [N, T_k, d_v].
        key_masks: A 2d tensor with shape of [N, key_seqlen]
        causality: If True, applies masking for future blinding
        dropout_rate: A floating point number of [0, 1].
        training: boolean for controlling droput
        scope: Optional scope for `variable_scope`.
        """
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            d_k = Q.get_shape().as_list()[-1]

            # dot product
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale
            outputs /= (d_k ** 0.5)

            # softmax
            alpha = tf.nn.softmax(outputs)
            # visualize attention
            attention = tf.transpose(alpha, [0, 2, 1])
            attention_sum = tf.reduce_sum(attention, -1)
            # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # weighted sum (context vectors)
            outputs = tf.matmul(alpha, V)  # (N, T_q, d_v)
        return outputs, attention_sum

    """-----------------------------------------HAI and HAI_rollout------------------------------------------"""

    def HAI(self, is_training, initializer, name='HAI'):
        # Extract some relevant config keys for convenience
        dims_base = cfg.DIMS_BASE

        # Input placeholders
        # dims: batch-time-height-width-channel
        self._rl_in = tf.placeholder(tf.float32, shape=[None, None, None, dims_base])
        self._rl_hid = tf.placeholder(tf.float32, shape=[None, None, None, 300])

        # Define convenience operator
        self.conv = lambda i, k: tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')

        # Create scaled dot-product attention's kernals
        self.wq_kernal = self._make_var('wq_weights', [3, 3, 300, dims_base],
                                        initializer, is_training)
        self.wk_kernal = self._make_var('wk_weights', [3, 3, dims_base, dims_base],
                                        initializer, is_training)
        self.wv_kernal = self._make_var('wv_weights', [3, 3, dims_base, dims_base],
                                        initializer, is_training)

        # Create conv-GRU kernels
        self.xr_kernel_base = self._make_var('xr_weights_base', [3, 3, dims_base, 300],
                                             initializer, is_training)
        self.xh_kernel_base = self._make_var('xh_weights_base', [3, 3, dims_base, 300],
                                             initializer, is_training)
        self.xz_kernel_base = self._make_var('xz_weights_base', [3, 3, dims_base, 300],
                                             initializer, is_training)
        self.hr_kernel = self._make_var('hr_weights', [3, 3, 300, 300],
                                        initializer, is_training)
        self.hh_kernel = self._make_var('hh_weights', [3, 3, 300, 300],
                                        initializer, is_training)
        self.hz_kernel = self._make_var('hz_weights', [3, 3, 300, 300],
                                        initializer, is_training)
        self.h_relu_kernel = self._make_var('h_relu_weights', [3, 3, 300, 128],
                                            initializer, is_training)

        # Create Conv-GRU biases
        bias_init = initializer
        self.r_bias = self._make_var('r_bias', [300], bias_init, is_training)
        self.h_bias = self._make_var('h_bias', [300], bias_init, is_training)
        self.z_bias = self._make_var('z_bias', [300], bias_init, is_training)
        self.relu_bias = self._make_var('relu_bias', [128], bias_init, is_training)

        # Used for some aux info (e.g. exploration penalty when used as feature)
        add_dim = 128
        self.additional_kernel = self._make_var('additional_weights', [3, 3, add_dim, 1],
                                                initializer, is_training)
        self.additional_bias = self._make_var('additional_bias', [1], bias_init,
                                              is_training)

        # Define weights for stopping condition (no bias here)
        self.done_weights = self._make_var('done_weights', [625, 1], initializer,
                                           is_training)

        # We need to make a TensorFlow dynamic graph-style while-loop, as our
        # conv-GRU will be unrolled a different number of steps depending on the
        # termination decisions of the agent

        # First we need to set some init / dummy variables
        in_shape = tf.shape(self._rl_in)
        done_logits_all = tf.zeros([0, 1])
        fix_logits_all = tf.zeros([0, in_shape[1] * in_shape[2]])
        done_prob = tf.zeros([0, 1])
        fix_prob_map = tf.zeros([0, 0, 0, 0])
        h = tf.slice(self._rl_hid, [0, 0, 0, 0], [1, -1, -1, -1])

        # Looping termination condition (TF syntax demands also the other variables
        # are sent as input, although not used for the condition check)
        nbr_steps = in_shape[0]
        i = tf.constant(0)
        while_cond = lambda i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map: tf.less(i, nbr_steps)

        # Unroll current step (if forward pass) and if in training unroll
        # a full rollout
        i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map \
            = tf.while_loop(while_cond, self.HAI_rollout,
                            [i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map],
                            shape_invariants=[i.get_shape(), tf.TensorShape([None, 1]),
                                              tf.TensorShape([None, None]), tf.TensorShape([None, 1]),
                                              h.get_shape(), tf.TensorShape([None, None, None, None])])

        # Insert to containers
        self._predictions['done_prob'] = done_prob
        self._predictions['fix_prob'] = fix_prob_map
        self._predictions['done_logits'] = done_logits_all
        self._predictions['fix_logits'] = fix_logits_all
        self._predictions['rl_hid'] = h

    # Unroll the HAI
    def HAI_rollout(self, i, done_logits_all, fix_logits_all, done_prob, h, fix_prob_map):

        # Extract some relevant config keys for convenience
        dims_base = cfg.DIMS_BASE

        # Split into base feature map and auxiliary input
        rl_base = tf.slice(self._rl_in, [i, 0, 0, 0], [1, -1, -1, dims_base])

        input_shape = tf.shape(rl_base)
        height = input_shape[1]
        width = input_shape[2]

        Q = self.conv(h, self.wq_kernal)
        K = self.conv(rl_base, self.wk_kernal)
        V = self.conv(rl_base, self.wv_kernal)
        Q_flatten = tf.reshape(Q, [1, -1, dims_base])
        K_flatten = tf.reshape(K, [1, -1, dims_base])
        V_flatten = tf.reshape(V, [1, -1, dims_base])
        attention_outputs, attention_sum = self.scaled_dot_product_attention_1(Q_flatten, K_flatten, V_flatten)
        attention_outputs_reshape = tf.reshape(attention_outputs, tf.shape(rl_base))
        attention_sum = tf.reshape(attention_sum, [1, height, width, 1])

        # eq. (1)
        # size=(1,h,w,300)
        xr_conv = self.conv(attention_outputs_reshape, self.xr_kernel_base)
        # size=(1,h,w,300)
        hr_conv = self.conv(h, self.hr_kernel)
        r = tf.sigmoid(xr_conv + hr_conv + self.r_bias)

        # eq. (2)
        # size=(1,h,w,300)
        xh_conv = self.conv(attention_outputs_reshape, self.xh_kernel_base)
        # size=(1,h,w,300)
        hh_conv = self.conv(r * h, self.hh_kernel)
        hbar = tf.tanh(xh_conv + hh_conv + self.h_bias)

        # eq. (3)
        # size=(1,h,w,300)
        xz_conv = self.conv(attention_outputs_reshape, self.xz_kernel_base)
        # size=(1,h,w,300)
        hz_conv = self.conv(h, self.hz_kernel)
        z = tf.sigmoid(xz_conv + hz_conv + self.z_bias)

        # eq. (4), Ht
        # size=(1,h,w,300)
        h = (1 - z) * h + z * hbar

        # eq. (5), A^t
        # size=(1,h,w,128)
        conv_gru = tf.nn.relu(self.conv(h, self.h_relu_kernel) + self.relu_bias)

        # Extract relevant stuff
        batch_sz = 1  # must be 1

        # eq. (6), At
        conv_gru_processed = tf.nn.tanh(self.conv(conv_gru, self.additional_kernel) \
                                        + self.additional_bias)

        # done_slice = tf.slice(conv_gru_processed, [0, 0, 0, 0], [1, -1, -1, 1])
        done_slice = conv_gru_processed
        # fix_slice = tf.slice(conv_gru_processed, [0, 0, 0, 1], [1, -1, -1, 1])
        done_slice_reshaped = tf.image.resize_images(done_slice, [25, 25])
        done_slice_vecd = tf.reshape(done_slice_reshaped, [batch_sz, 625])
        done_logits = tf.matmul(done_slice_vecd, self.done_weights)
        done_prob = tf.sigmoid(done_logits)  # size=(?,1)

        reshape_layer = tf.reshape(tf.transpose(attention_sum, [0, 3, 1, 2]),
                                   [1, 1, height * width])
        smax_layer = tf.nn.softmax(reshape_layer)
        fix_prob_map = tf.transpose(tf.reshape(smax_layer,
                                               [1, 1, height, width]), [0, 2, 3, 1])
        fix_slice_logits = tf.reshape(attention_sum, [batch_sz, -1])

        # Append
        done_logits_all = tf.concat([done_logits_all, done_logits], 0)
        fix_logits_all = tf.concat([fix_logits_all, fix_slice_logits], 0)

        # Return
        return tf.add(i, 1), done_logits_all, fix_logits_all, done_prob, h, fix_prob_map

    """---------------------------------------------------------------------------------------"""

    def _initial_rl_input(self, net_conv, anchors, rpn_bbox_pred, name='rl_in_init'):

        # Form initial input
        shape_info = tf.shape(net_conv)
        batch_sz = shape_info[0]
        height = shape_info[1]
        width = shape_info[2]
        rl_in_init = net_conv / tf.reduce_max(net_conv)

        self._predictions['rl_in_init'] = rl_in_init

        roi_obs_vol = tf.zeros((batch_sz, height, width, cfg.NBR_ANCHORS),
                               dtype=tf.int32)
        self._predictions['roi_obs_vol'] = roi_obs_vol

        self._predictions['rois_fixate'] = self._proposal_layer_fixate(anchors, rpn_bbox_pred)

    def get_init_rl(self, sess, image, im_info):
        feed_dict = {self._image: image, self._im_info: im_info,
                     self._net_conv_in: np.zeros((1, 1, 1, cfg.DIMS_BASE))}
        net_conv, rl_in, roi_obs_vol \
            = sess.run([self._predictions['net_conv'], self._predictions['rl_in_init'],
                        self._predictions['roi_obs_vol']], feed_dict=feed_dict)
        # Create LRP-HAI hidden state (conv-GRU hidden state)
        batch_sz, height, width = rl_in.shape[:3]
        # rl_hid = np.zeros((batch_sz, height, width, 300))

        """----------------------ATTENTION MODULE'S HIDDEN STATE--------------------"""
        rl_hid = np.zeros((batch_sz, height, width, 300))
        """-------------------------------------------------------------------------"""

        # Get height, width of downsized feature map and orig. feature map, and also
        # calculate the fixation rectangle size
        height, width = rl_in.shape[1:3]
        fix_rect_h = int(round(cfg.LRP_HAI.H_FIXRECT * height))
        fix_rect_w = int(round(cfg.LRP_HAI.W_FIXRECT * width))

        # Return
        return net_conv, rl_in, rl_hid, roi_obs_vol, \
               height, width, fix_rect_h, fix_rect_w

    def pass_to_rpn(self, sess, fixate_region, h_start, w_start, im_info):
        h, w = fixate_region.shape[1:3]
        feed_dict_rpn = {self._net_conv_fixate: fixate_region,
                         self.fix_rect_h: h,
                         self.fix_rect_w: w,
                         self.h_start: h_start,
                         self.w_start: w_start, self._im_info: im_info}

        rois_fixate = sess.run(self._predictions['rois_fixate'], feed_dict=feed_dict_rpn)

        return rois_fixate

    def action_pass(self, sess, rl_in, rl_hid, is_training=True):
        """ This is the "forward pass" of the LRP-HAI action selection """

        if is_training:
            # Store in containers (used backpropagating gradients)

            # must copy rl_in -- otherwise rl-updates in future affect past states
            # due to 'pass-by-reference' in python numpy-array-updates
            self._ep['x'].append(np.copy(rl_in))
            self._ep['h'].append(rl_hid)
        feed_dict_action = {self._rl_in: rl_in, self._rl_hid: rl_hid}
        rl_hid, done_prob, fix_prob \
            = sess.run([self._predictions['rl_hid'],
                        self._predictions['done_prob'],
                        self._predictions['fix_prob'],
                        ], feed_dict=feed_dict_action)

        fix_prob = fix_prob[0, :, :, 0]
        return rl_hid, done_prob, fix_prob

    def seq_rois_pass(self, sess, net_conv, rois_seq, im_info, is_train_det=False):
        """
    This function handles the per-fixation sequential forwarding of RoIs
    for class-specific predictions
    """
        feed_dict_seq = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                         self._cond_switch_roi: 1, self._gt_boxes: np.zeros((1, 5)), self._im_info: im_info}
        cls_prob_seq, bbox_preds_seq = sess.run([self._predictions['cls_prob_seq'],
                                                 self._predictions['bbox_pred_seq']],
                                                feed_dict=feed_dict_seq)
        # If test-time (or any time, e.g. LRP-HAI training, where we are NOT
        # specifically training the detector component), need to "undo"
        # mean-std normalization
        if not is_train_det:
            bbox_preds_seq *= cfg.STDS_BBOX
            bbox_preds_seq += cfg.MEANS_BBOX
        return cls_prob_seq, bbox_preds_seq

    ############# LRP-HAI ADDITIONAL COMPONENTS -- END ##########################

    def _image_to_head(self, is_training, reuse=None):
        raise NotImplementedError

    def _head_to_tail(self, pool5, is_training, reuse=None):
        raise NotImplementedError

    def create_architecture(self, mode, num_classes, tag=None,
                            anchor_sizes=(128, 256, 512), anchor_strides=(16,), anchor_ratios=(0.5, 1, 2)):
        self._image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        self._im_info = tf.placeholder(tf.float32, shape=[3])
        self._gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self._tag = tag

        self._num_classes = num_classes
        self._mode = mode

        self._anchor_sizes = anchor_sizes
        self._num_sizes = len(anchor_sizes)
        self._anchor_strides = anchor_strides
        self._num_strides = len(anchor_strides)
        self._anchor_ratios = anchor_ratios
        self._num_ratios = len(anchor_ratios)
        self._num_anchors = self._num_sizes * self._num_ratios

        training = mode == 'TRAIN'

        # handle most of the regularizers here
        weights_regularizer \
            = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY)
        if cfg.TRAIN.BIAS_DECAY:
            biases_regularizer = weights_regularizer
        else:
            biases_regularizer = tf.no_regularizer

        # list as many types of layers as possible, even if they are not used now
        with arg_scope([slim.conv2d, slim.conv2d_in_plane,
                        slim.conv2d_transpose, slim.separable_conv2d,
                        slim.fully_connected],
                       weights_regularizer=weights_regularizer,
                       biases_regularizer=biases_regularizer,
                       biases_initializer=tf.constant_initializer(0.0)):
            self._build_network(training)

        layers_to_output = {}
        if training:
            self._add_losses()
            layers_to_output.update(self._losses)
        layers_to_output.update(self._predictions)
        self.fr_tvars = tf.trainable_variables()

        return layers_to_output

    def get_variables_to_restore(self, variables, var_keep_dic):
        raise NotImplementedError

    def fix_variables(self, sess, pretrained_model, do_reverse):
        raise NotImplementedError

    # Extract the head feature maps, for example for vgg16 it is conv5_3
    # only useful during testing mode
    def extract_head(self, sess, image):
        feed_dict = {self._image: image}
        feat = sess.run(self._layers["head"], feed_dict=feed_dict)
        return feat

    # only useful during testing mode
    def test_image(self, sess, image, im_info):
        feed_dict = {self._image: image, self._im_info: im_info}
        cls_score, cls_prob, bbox_pred, rois \
            = sess.run([self._predictions["cls_score_seq"],
                        self._predictions['cls_prob_seq'],
                        self._predictions['bbox_pred_seq'], self._predictions['rois']],
                       feed_dict=feed_dict)
        return cls_score, cls_prob, bbox_pred, rois

    def train_step_det(self, sess, train_op, net_conv, rois_seq, gt_boxes,
                       im_info):
        feed_dict = {self._net_conv_in: net_conv, self._rois_seq: rois_seq,
                     self._im_info: im_info, self._gt_boxes: gt_boxes,
                     self._image: np.zeros((1, 1, 1, 3)), self._cond_switch_roi: 0}
        loss_cls, loss_box, loss, _ \
            = sess.run([self._losses['cross_entropy'], self._losses['loss_box'],
                        self._losses['total_loss'], train_op], feed_dict=feed_dict)
        return loss_cls, loss_box, loss

    def train_step_no_return(self, sess, blobs, train_op):
        feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
                     self._gt_boxes: blobs['gt_boxes']}
        sess.run([train_op], feed_dict=feed_dict)
