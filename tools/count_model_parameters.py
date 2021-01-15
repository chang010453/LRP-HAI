from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths_drl
import time, os, sys
import numpy as np
from utils.logger import setup_logger

import tensorflow as tf


logger = setup_logger("LRP-HAI", save_dir="output", filename="log_parameters.txt")
if len(sys.argv) == 2:
    ckpt_fpath = sys.argv[1]
else:
    logger.info('Usage: python count_ckpt_param.py path-to-ckpt')
    sys.exit(1)

# Open TensorFlow ckpt
reader = tf.train.NewCheckpointReader(ckpt_fpath)

logger.info('\nCount the number of parameters in ckpt file(%s)' % ckpt_fpath)
param_map = reader.get_variable_to_shape_map()
total_count = 0
for k, v in param_map.items():
    if 'Momentum' not in k and 'global_step' not in k:
        temp = np.prod(v)
        total_count += temp
        logger.info('%s: %s => %d' % (k, str(v), temp))

logger.info('Total Param Count: %d' % total_count)
