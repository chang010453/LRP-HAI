# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Partially written* by Chang Hsiao-Chien
# Partially Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import os
import sys
import glob
import time
from time import sleep

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

from model.config import cfg, cfg_from_list
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
from utils.statcoll import StatCollector
from model.factory import run_LRP_HAI
import logging


class SolverWrapper(object):
    """ A wrapper class for the training process """

    def __init__(self, sess, network, imdb, roidb, valroidb, output_dir,
                 pretrained_model=None, logger=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.valroidb = valroidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.logger = logger

    def snapshot(self, sess, iter):
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Store the model snapshot
        filename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.ckpt'
        filename = os.path.join(self.output_dir, filename)
        self.saver.save(sess, filename)
        self.logger.info('Wrote snapshot to: {:s}'.format(filename))

        # Also store some meta information, random state, etc.
        nfilename = cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}'.format(iter) + '.pkl'
        nfilename = os.path.join(self.output_dir, nfilename)
        # current state of numpy random
        st0 = np.random.get_state()
        # current position in the database
        cur = self.data_layer._cur
        # current shuffled indexes of the database
        perm = self.data_layer._perm
        # current position in the validation database
        cur_val = self.data_layer_val._cur
        # current shuffled indexes of the validation database
        perm_val = self.data_layer_val._perm

        # Dump the meta info
        with open(nfilename, 'wb') as fid:
            pickle.dump(st0, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(cur_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(perm_val, fid, pickle.HIGHEST_PROTOCOL)
            pickle.dump(iter, fid, pickle.HIGHEST_PROTOCOL)

        return filename, nfilename

    def from_snapshot(self, sess, sfile, nfile):
        self.logger.info('Restoring model snapshots from {:s}'.format(sfile))
        self.saver.restore(sess, sfile)
        self.logger.info('Restored.')
        # Needs to restore the other hyper-parameters/states for training, I have
        # tried my best to find the random states so that it can be recovered exactly
        # However the Tensorflow state is currently not available
        with open(nfile, 'rb') as fid:
            st0 = pickle.load(fid)
            cur = pickle.load(fid)
            perm = pickle.load(fid)
            cur_val = pickle.load(fid)
            perm_val = pickle.load(fid)
            last_snapshot_iter = pickle.load(fid)

            np.random.set_state(st0)
            self.data_layer._cur = cur
            self.data_layer._perm = perm
            self.data_layer_val._cur = cur_val
            self.data_layer_val._perm = perm_val

        return last_snapshot_iter

    def get_variables_in_checkpoint_file(self, file_name):
        try:
            reader = pywrap_tensorflow.NewCheckpointReader(file_name)
            var_to_shape_map = reader.get_variable_to_shape_map()
            return var_to_shape_map
        except Exception as e:  # pylint: disable=broad-except
            print(str(e))
            if "corrupted compressed block contents" in str(e):
                print("It's likely that your checkpoint file has been compressed "
                      "with SNAPPY.")

    def _return_gradients(self, gvs):
        # grads, vars = gvs
        grads = [g for g, _ in gvs]
        vars = [v for _, v in gvs]
        return [grad if grad is not None else tf.zeros_like(var)
                for var, grad in zip(vars, grads)]

    def _compute_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        return [grad if grad is not None else tf.zeros_like(var)
                for var, grad in zip(var_list, grads)]

    def construct_graph(self, sess):
        # Set the random seed for tensorflow
        tf.set_random_seed(cfg.RNG_SEED)
        with sess.graph.as_default():
            # Build the main computation graph
            layers = self.net.create_architecture('TRAIN', self.imdb.num_classes, tag='default',
                                                  anchor_sizes=cfg.ANCHOR_SIZES,
                                                  anchor_strides=cfg.ANCHOR_STRIDES,
                                                  anchor_ratios=cfg.ANCHOR_RATIOS)
            # Define the loss
            loss = layers['total_loss']
            # Set learning rate and momentum
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            self.optimizer = tf.train.MomentumOptimizer(lr, cfg.TRAIN.MOMENTUM)

            # Compute the gradients with regard to the loss
            gvs = self.optimizer.compute_gradients(loss)
            self.net.grads = self._compute_gradients(loss, self.net.fr_tvars)
            self.return_grads = self._return_gradients(gvs)
            train_op = self.optimizer.apply_gradients(gvs)

            # Initialize main LRP-HAI network
            self.net.build_LRP_HAI_network()

        return lr, train_op

    def find_previous(self):
        sfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.ckpt.meta')
        sfiles = glob.glob(sfiles)
        sfiles.sort(key=os.path.getmtime)
        # Get the snapshot name in TensorFlow
        redfiles = []
        for stepsize in cfg.TRAIN.STEPSIZE:
            redfiles.append(os.path.join(self.output_dir,
                                         cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_{:d}.ckpt.meta'.format(stepsize + 1)))
        sfiles = [ss.replace('.meta', '') for ss in sfiles if ss not in redfiles]

        nfiles = os.path.join(self.output_dir, cfg.TRAIN.SNAPSHOT_PREFIX + '_iter_*.pkl')
        nfiles = glob.glob(nfiles)
        nfiles.sort(key=os.path.getmtime)
        redfiles = [redfile.replace('.ckpt.meta', '.pkl') for redfile in redfiles]
        nfiles = [nn for nn in nfiles if nn not in redfiles]

        lsf = len(sfiles)
        assert len(nfiles) == lsf

        return lsf, nfiles, sfiles

    def initialize(self, sess):
        # Initial file lists are empty
        np_paths = []
        ss_paths = []
        # Fresh train directly from ImageNet weights
        self.logger.info('Loading initial model weights from {:s}'.format(self.pretrained_model))
        variables = tf.global_variables()
        # Initialize all variables first
        sess.run(tf.variables_initializer(variables, name='init'))
        var_keep_dic = self.get_variables_in_checkpoint_file(self.pretrained_model)
        # print(self.pretrained_model)
        # sleep(100)
        # Get the variables to restore, ignoring the variables to fix
        variables_to_restore = self.net.get_variables_to_restore(variables,
                                                                 var_keep_dic)
        restorer = tf.train.Saver(variables_to_restore)
        restorer.restore(sess, self.pretrained_model)
        self.logger.info('Loaded.')

        last_snapshot_iter = 0
        fr_rate = cfg.TRAIN.LEARNING_RATE
        fr_stepsize = cfg.TRAIN.STEPSIZE
        drl_rate = cfg.LRP_HAI_TRAIN.LEARNING_RATE
        drl_stepsize = cfg.LRP_HAI_TRAIN.STEPSIZE
        return fr_rate, drl_rate, last_snapshot_iter, fr_stepsize, drl_stepsize, np_paths, ss_paths

    def restore(self, sess, sfile, nfile):
        # Get the most recent snapshot and restore
        np_paths = [nfile]
        ss_paths = [sfile]
        # Restore model from snapshots
        last_snapshot_iter = self.from_snapshot(sess, sfile, nfile)
        # Set the learning rate
        fr_rate = cfg.TRAIN.LEARNING_RATE
        fr_stepsize = cfg.TRAIN.STEPSIZE[0]
        drl_rate = cfg.LRP_HAI_TRAIN.LEARNING_RATE
        drl_stepsize = cfg.LRP_HAI_TRAIN.STEPSIZE
        if last_snapshot_iter > fr_stepsize:
            fr_rate *= cfg.TRAIN.GAMMA
        if last_snapshot_iter > drl_stepsize:
            drl_rate *= cfg.LRP_HAI_TRAIN.GAMMA

        return fr_rate, drl_rate, last_snapshot_iter, fr_stepsize, drl_stepsize, np_paths, ss_paths

    def remove_snapshot(self, np_paths, ss_paths):
        to_remove = len(np_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            nfile = np_paths[0]
            os.remove(str(nfile))
            np_paths.remove(nfile)
        to_remove = len(ss_paths) - cfg.TRAIN.SNAPSHOT_KEPT
        for c in range(to_remove):
            sfile = ss_paths[0]
            # To make the code compatible to earlier versions of Tensorflow,
            # where the naming tradition for checkpoints are different
            if os.path.exists(str(sfile)):
                os.remove(str(sfile))
            else:
                os.remove(str(sfile + '.data-00000-of-00001'))
                os.remove(str(sfile + '.index'))
            sfile_meta = sfile + '.meta'
            os.remove(str(sfile_meta))
            ss_paths.remove(sfile)

    def _print_det_loss(self, iter, max_iters, tot_loss, loss_cls, loss_box,
                        lr, timer, in_string='detector'):
        if (iter + 1) % (cfg.TRAIN.DISPLAY) == 0:
            if loss_box is not None:
                self.logger.info('iter: %d / %d, total loss: %.6f\n '
                                 '>>> loss_cls (%s): %.6f\n '
                                 '>>> loss_box (%s): %.6f\n >>> lr: %f' % \
                                 (iter + 1, max_iters, tot_loss, in_string, loss_cls, in_string,
                                  loss_box, lr))
            else:
                self.logger.info('iter: %d / %d, total loss (%s): %.6f\n >>> lr: %f' % \
                                 (iter + 1, max_iters, in_string, tot_loss, lr))
            self.logger.info('speed: {:.3f}s / iter'.format(timer.average_time))

    def _check_if_continue(self, iter, max_iters, snapshot_add):
        img_start_idx = cfg.LRP_HAI_TRAIN.IMG_START_IDX
        if iter > img_start_idx:
            return iter, max_iters, snapshot_add, False
        if iter < img_start_idx:
            self.logger.info("iter %d < img_start_idx %d -- continuing" % (iter, img_start_idx))
            iter += 1
            return iter, max_iters, snapshot_add, True
        if iter == img_start_idx:
            self.logger.info("Adjusting stepsize, train-det-start etcetera")
            snapshot_add = img_start_idx
            max_iters -= img_start_idx
            iter = 0
            cfg_from_list(['LRP_HAI_TRAIN.IMG_START_IDX', -1])
            cfg_from_list(['LRP_HAI_TRAIN.DET_START',
                           cfg.LRP_HAI_TRAIN.DET_START - img_start_idx])
            cfg_from_list(['LRP_HAI_TRAIN.STEPSIZE',
                           cfg.LRP_HAI_TRAIN.STEPSIZE - img_start_idx])
            cfg_from_list(['TRAIN.STEPSIZE', [cfg.TRAIN.STEPSIZE[0] - img_start_idx]])
            self.logger.info("Done adjusting stepsize, train-det-start etcetera")
            return iter, max_iters, snapshot_add, False

    def train_model(self, sess, max_iters):
        # Build data layers for both training and validation set
        self.data_layer = RoIDataLayer(self.roidb, cfg.NBR_CLASSES)
        self.data_layer_val = RoIDataLayer(self.valroidb, cfg.NBR_CLASSES, True)

        # Construct the computation graph corresponding to the original Faster R-CNN
        # architecture first
        lr_det_op, train_op = self.construct_graph(sess)

        # We will handle the snapshots ourselves
        self.saver = tf.train.Saver(max_to_keep=100000)

        # Find previous snapshots if there is any to restore from
        lsf, nfiles, sfiles = self.find_previous()

        # Initialize the variables or restore them from the last snapshot
        if lsf == 0:
            fr_rate, drl_rate, last_snapshot_iter, \
            fr_stepsize, drl_stepsize, np_paths, ss_paths = self.initialize(sess)
        else:
            fr_rate, drl_rate, last_snapshot_iter, \
            fr_stepsize, drl_stepsize, np_paths, ss_paths = self.restore(sess,
                                                                         str(sfiles[-1]),
                                                                         str(nfiles[-1]))

        # Initialize
        self.net.init_rl_train(sess)

        # Setup initial learning rates
        # 0.00002
        lr_rl = drl_rate
        # 0.00025
        lr_det = fr_rate
        sess.run(tf.assign(lr_det_op, lr_det))

        # Sample first beta
        beta = cfg.LRP_HAI_TRAIN.BETA

        # Setup LRP-HAI timers
        timers = {'init': Timer(), 'fulltraj': Timer(), 'upd-obs-vol': Timer(),
                  'upd-seq': Timer(), 'upd-rl': Timer(), 'action-rl': Timer(),
                  'coll-traj': Timer(), 'run-LRP-HAI': Timer(),
                  'train-LRP-HAI': Timer(), 'batch_time': Timer(), 'total': Timer()}

        # Create StatCollector (tracks various RL training statistics)
        stat_strings = ['rews_total_traj', 'traj-len', 'frac-area',
                        'gt >= 0.5 frac', 'gt-IoU-frac']
        sc = StatCollector(max_iters, stat_strings, cfg.LRP_HAI_TRAIN.BATCH_SIZE, self.output_dir)

        timer = Timer()
        iter = last_snapshot_iter
        snapshot_add = 0
        timers['total'].tic()
        timers['batch_time'].tic()
        while iter < max_iters:
            # Get training data, one batch at a time (assumes batch size 1)
            blobs = self.data_layer.forward()

            iter, max_iters, snapshot_add, do_continue \
                = self._check_if_continue(iter, max_iters, snapshot_add)
            if do_continue:
                continue
            # Potentially update LRP-HAI learning rate
            # 90000
            if (iter + 1) % cfg.LRP_HAI_TRAIN.STEPSIZE == 0:
                # lr_rl = lr_rl * 0.2
                lr_rl *= cfg.LRP_HAI_TRAIN.GAMMA

            # Run LRP-HAI in training mode
            timers['run-LRP-HAI'].tic()
            net_conv, rois_LRP_HAI, gt_boxes, im_info, timers, stats = run_LRP_HAI(sess, self.net, blobs, timers, mode='train',
                                                                           beta=beta, im_idx=None, extra_args=lr_rl,
                                                                           alpha=cfg.LRP_HAI.ALPHA)
            timers['run-LRP-HAI'].toc()

            # BATCH_SIZE = 50
            if (iter + 1) % cfg.LRP_HAI_TRAIN.BATCH_SIZE == 0:
                self.logger.info("\n##### LRP-HAI BATCH GRADIENT UPDATE - START ##### \n")
                self.logger.info('iter: %d / %d' % (iter + 1, max_iters))
                self.logger.info('lr-rl: %f' % lr_rl)
                timers['train-LRP-HAI'].tic()
                self.net.train_LRP_HAI(sess, lr_rl, sc, stats)
                timers['train-LRP-HAI'].toc()
                sc.print_stats(iter=iter + 1, logger=self.logger)

                batch_time = timers['batch_time'].toc()
                self.logger.info('TIMINGS:')
                self.logger.info('runnn-LRP-HAI: %.4f' % timers['run-LRP-HAI'].get_avg())
                self.logger.info('train-LRP-HAI: %.4f' % timers['train-LRP-HAI'].get_avg())
                self.logger.info('train-LRP-HAI-batch: %.4f' % batch_time)
                self.logger.info("\n##### LRP-HAI BATCH GRADIENT UPDATE - DONE ###### \n")

                timers['batch_time'].tic()
            else:
                sc.update(0, 0, 0, stats)

            # At this point we assume that an RL-trajectory has been performed.
            # We next train detector with LRP-HAI running in deterministic mode.
            # Potentially train detector component of network
            # DET_START = 40000
            if 0 <= cfg.LRP_HAI_TRAIN.DET_START <= iter:

                # Run LRP-HAI in deterministic mode
                # net_conv, rois_LRP_HAI, gt_boxes, im_info, timers \
                #     = run_LRP_HAI(sess, self.net, blobs, timers, mode='train_det',
                #                   beta=beta, im_idx=None, alpha=cfg.LRP_HAI.ALPHA)

                # Learning rate
                if (iter + 1) % cfg.TRAIN.STEPSIZE[0] == 0:
                    lr_det *= cfg.TRAIN.GAMMA
                    sess.run(tf.assign(lr_det_op, lr_det))

                if rois_LRP_HAI is not None:
                    timer.tic()
                    # Train detector part
                    loss_cls, loss_box, tot_loss \
                        = self.net.train_step_det(sess, train_op, net_conv, rois_LRP_HAI,
                                                gt_boxes, im_info)
                    timer.toc()

                # Display training information
                self._print_det_loss(iter, max_iters, tot_loss, loss_cls, loss_box,
                                     lr_det, timer)

            # Snapshotting
            if (iter + 1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter + 1
                ss_path, np_path = self.snapshot(sess, iter + 1 + snapshot_add)
                np_paths.append(np_path)
                ss_paths.append(ss_path)

                # Remove the old snapshots if there are too many
                if len(np_paths) > cfg.TRAIN.SNAPSHOT_KEPT:
                    self.remove_snapshot(np_paths, ss_paths)

            # Increase iteration counter
            iter += 1

        # Potentially save one last time
        if last_snapshot_iter != iter:
            self.snapshot(sess, iter + snapshot_add)

        timers['total'].toc()
        total_time = timers['total'].total_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.logger.info("total time: %02d:%02d:%02d" % (h, m, s))


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')
    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')
    return imdb.roidb


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after))
    return filtered_roidb


def train_net(network, imdb, roidb, valroidb, output_dir,
              pretrained_model=None, max_iters=40000):
    """Train LRP-HAI for a Faster R-CNN network."""
    roidb = filter_roidb(roidb)
    valroidb = filter_roidb(valroidb)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    logger = logging.getLogger("LRP-HAI.train_net")

    with tf.Session(config=tfconfig) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, valroidb, output_dir,
                           pretrained_model, logger)
        logger.info('Solving...')
        sw.train_model(sess, max_iters)
        logger.info('done solving')
