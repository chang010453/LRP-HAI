# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Partially written by Chang Hsiao-Chien
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_drl
from model.factory import run_LRP_HAI, print_timings, get_image_blob
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
import numpy as np
from time import sleep
from utils.logger import setup_logger
from utils.timer import Timer
import cv2
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.P4 import P4


def parse_args():
    """
  Parse input arguments
  """

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='Test a LRP-HAI network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default='experiments/cfgs/LRP-HAI-P4.yml', type=str)
    parser.add_argument('--model', dest='model',
                        help='model to test',
                        default='/media/data/LRP-HAI/experiments/drl-model-2/P4/res101/drl-model-2-1/output/res101_LRP_HAI/cell_train/res101_LRP_HAI_iter_110000.ckpt',
                        type=str)

    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        # default='voc_2007_test', type=str)
                        default='cell_val', type=str)

    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
    parser.add_argument('--nbr_fix', dest='nbr_fix',
                        help='0: auto-stop, > 0 run LRP-HAI exactly nbr_fix steps',
                        default=0, type=int)
    parser.add_argument("--alpha", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Activate alpha mode.")
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def im_detect(sess, net, im, timers, im_idx=None):
    # Setup image blob
    blobs = {}
    blobs['data'], im_scales, blobs['im_shape_orig'] = get_image_blob(im)
    im_blob = blobs['data']
    blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]])

    # Run LRP-HAI
    scores, pred_bboxes, timers, _ \
        = run_LRP_HAI(sess, net, blobs, timers, 'test', cfg.LRP_HAI_TEST.BETA,
                      im_idx, alpha=cfg.LRP_HAI.ALPHA)

    return scores, pred_bboxes, timers, _


if __name__ == '__main__':
    args = parse_args()

    # print('Called with args:')
    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('Using config:')
    # pprint.pprint(cfg)

    # if has model, get the name from it
    # if does not, then just use the initialization weights
    if args.model:
        filename = os.path.splitext(os.path.basename(args.model))[0]
    else:
        filename = os.path.splitext(os.path.basename(args.weight))[0]

    tag = args.tag
    tag = tag if tag else 'default'
    filename = tag + '/' + filename

    # This extra_string used by me (Aleksis) when running code on two
    # different machines, for convenience
    extra_string = ''
    # print(args.imdb_name)
    if args.imdb_name == 'voc_2012_test':
        extra_string += '_test'
    # print(args.imdb_name + extra_string)
    # sleep(100)
    imdb = get_imdb(args.imdb_name + extra_string)
    imdb.competition_mode(args.comp_mode)

    # new add
    output_dir = get_output_dir(imdb, filename)
    logger = setup_logger("LRP-HAI", save_dir=output_dir, filename="log_test.txt")
    logger.info('Called with args:')
    logger.info(args)

    # Set class names in config file based on IMDB
    class_names = imdb.classes
    cfg_from_list(['CLASS_NAMES', [class_names]])
    if args.alpha:
        cfg_from_list(['LRP_HAI.ALPHA', True])
    else:
        cfg_from_list(['LRP_HAI.ALPHA', False])

    # VISUALIZE IMAGE
    cfg_from_list(['LRP_HAI_TEST.DO_VISUALIZE', True])

    # Specify if run LRP-HAI in auto mode or a fix number of iterations
    cfg_from_list(['LRP_HAI_TEST.NBR_FIX', args.nbr_fix])
    logger.info('Using config:\n{}' .format(pprint.pformat(cfg)))
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    # tfconfig.gpu_options.visible_device_list = '0'
    # 最多占gpu资源的70%
    # tfconfig.gpu_options.per_process_gpu_memory_fraction = 0.7

    # Set the random seed for tensorflow
    tf.set_random_seed(cfg.RNG_SEED)

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if cfg.P4:
        # load network
        if args.net == 'res50':
            net = P4(num_layers=50)
        elif args.net == 'res101':
            net = P4(num_layers=101)
        else:
            raise NotImplementedError
    else:
        # load network
        if args.net == 'vgg16':
            net = vgg16()
        elif args.net == 'res50':
            net = resnetv1(num_layers=50)
        elif args.net == 'res101':
            net = resnetv1(num_layers=101)
        else:
            raise NotImplementedError

    # load model
    net.create_architecture("TEST", imdb.num_classes, tag='default',
                            anchor_sizes=cfg.ANCHOR_SIZES,
                            anchor_strides=cfg.ANCHOR_STRIDES,
                            anchor_ratios=cfg.ANCHOR_RATIOS)

    net.build_LRP_HAI_network(False)

    if args.model:
        logger.info('Loading model check point from {:s}'.format(args.model))
        tf.train.Saver().restore(sess, args.model)
        logger.info('Loaded.')
    else:
        logger.info('Loading initial weights from {:s}'.format(args.weight))
        sess.run(tf.global_variables_initializer())
        logger.info('Loaded.')

    # demo
    # Visualize search trajectories?
    do_visualize = cfg.LRP_HAI_TEST.DO_VISUALIZE

    load_dir = "demo_image"
    imgs = os.listdir(load_dir)
    _t = {'im_detect': Timer(), 'misc': Timer(), 'total_time': Timer()}
    _t_LRP_HAI = {'init': Timer(), 'fulltraj': Timer(),
                  'upd-obs-vol': Timer(), 'upd-seq': Timer(), 'upd-rl': Timer(),
                  'action-rl': Timer(), 'coll-traj': Timer()}
    for i in range(len(imgs)):

        # Need to know image index if performing visualizations
        if do_visualize:
            im_idx = i
        else:
            im_idx = None

        # Detect!
        im = cv2.imread(os.path.join(load_dir, imgs[i]))
        _t['im_detect'].tic()
        scores, boxes, _t_LRP_HAI, _ = im_detect(sess, net, im, _t_LRP_HAI,
                                                     im_idx)
        _t['im_detect'].toc()

        logger.info('\nim_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(imgs), _t['im_detect'].average_time))
        # print_timings(_t_LRP_HAI) # uncomment for some timing details!

    sess.close()