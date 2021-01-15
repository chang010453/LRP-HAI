# --------------------------------------------------------
# Tensorflow LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Written by Chang Hsiao-Chien
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_drl
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
from time import sleep
from utils.logger import setup_logger

import tensorflow as tf
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

    parser = argparse.ArgumentParser(description='Train a LRP-HAI network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='experiments/cfgs/LRP-HAI-P4.yml', type=str)
    parser.add_argument('--weight', dest='weight',
                        help='initialize with pretrained model weights',
                        default='/media/data/LRP-HAI/fr-rcnn-weights/P4/res101/cell_train/default/res101_faster_rcnn_iter_180000.ckpt',
                        type=str)
    # default = '/media/data/LRP-HAI/fr-rcnn-voc2007-2012-trainval/vgg16_faster_rcnn_iter_180000.ckpt'

    parser.add_argument('--save', dest='save_path',
                        help='path for saving model weights',
                        default='/media/data/LRP-HAI/experiment/drl-model-2/cell/P4/res101/print_loss/',
                        type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='cell_train', type=str)
                        # default='voc_2007_trainval+voc_2012_trainval', type=str)
    parser.add_argument('--imdbval', dest='imdbval_name',
                        help='dataset to validate on',
                        default='cell_val', type=str)
                        # default='voc_2007_test', type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=110000, type=int)
    parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152, mobile',
                        default='res101', type=str)
    parser.add_argument('--det_start', dest='det_start',
                        help='-1: dont train detector; >=0: train detector onwards',
                        default=40000, type=int)
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


def combined_roidb(imdb_names):
    """
  Combine multiple roidbs
  """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':
    args = parse_args()

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    np.random.seed(cfg.RNG_SEED)

    # train set
    imdb, roidb = combined_roidb(args.imdb_name)

    # Set class names in config file based on IMDB
    class_names = imdb.classes
    cfg_from_list(['CLASS_NAMES', [class_names]])

    if args.alpha:
        cfg_from_list(['LRP_HAI.ALPHA', True])

    # Update config to match start of training detector
    cfg_from_list(['LRP_HAI_TRAIN.DET_START', args.det_start])

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, args.tag, args.save_path)

    logger = setup_logger("LRP-HAI", save_dir=args.save_path, filename="log_train.txt")
    logger.info('Called with args:')
    logger.info(args)
    logger.info('Using attention alpha:')
    logger.info(cfg.LRP_HAI.ALPHA)
    logger.info('Using config:\n{}'.format(pprint.pformat(cfg)))
    logger.info('{:d} roidb entries'.format(len(roidb)))
    logger.info('Output will be saved to `{:s}`'.format(output_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = False
    _, valroidb = combined_roidb(args.imdbval_name)
    logger.info('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

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

    train_net(net, imdb, roidb, valroidb, output_dir, pretrained_model=args.weight,
              max_iters=args.max_iters)
