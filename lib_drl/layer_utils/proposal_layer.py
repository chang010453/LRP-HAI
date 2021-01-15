# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms


# --------------------------------------------------------
# fixate proposals for LRP-HAI. Written by Chang Hsiao-Chien.
# Licensed under The MIT License [see LICENSE for details]
# Originally written by Ross Girshick and Xinlei Chen.
# --------------------------------------------------------
def proposal_layer_fixate(rpn_bbox_pred, im_info, anchors):
    """
    Simply returns every single RoI; LRP-HAI later decides
    which are forwarded to the class-specific module.
    """

    # Get the bounding boxes
    batch_sz, height, width = rpn_bbox_pred.shape[0: 3]
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    proposals = clip_boxes(proposals, im_info[:2])

    # Create initial (all-zeros) observation RoI volume
    roi_obs_vol = np.zeros((batch_sz, height, width, cfg.NBR_ANCHORS),
                           dtype=np.int32)

    not_keep_ids = np.zeros((1, 1), dtype=np.int32)

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    rois_all = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return rois_all, roi_obs_vol, not_keep_ids
