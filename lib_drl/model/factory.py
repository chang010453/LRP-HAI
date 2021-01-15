# --------------------------------------------------------
# Factory file for LRP-HAI
# Licensed under The MIT License [see LICENSE for details]
# Written by Chang Hsiao-Chien
# Written by Aleksis Pirinen
# Faster R-CNN code by Zheqi he, Xinlei Chen, based on code
# from Ross Girshick
# --------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
from time import sleep

from skimage.transform import resize as resize

from scipy.misc import imsave as imsave
from scipy.misc import imread as imread
from scipy.spatial.distance import cdist as cdist

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.blob import im_list_to_blob

from model.config import cfg
from model.nms_wrapper import nms
from model.bbox_transform import bbox_transform_inv, clip_boxes


def get_image_blob(im):
    """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors), im_shape


def IoU(bboxes, gt_bboxes):
    # If background is included in bboxes, remove it
    if bboxes.shape[1] > 4 * cfg.NBR_CLASSES:
        bboxes = bboxes[:, 4:]

    # Get some relevant sizes and pre-allocate memory
    nbr_bboxes = bboxes.shape[0]
    tot_nbr_gt_bboxes = gt_bboxes.shape[0]

    # Now we can pre-allocate the memory needed
    bbox_max_ious = np.zeros((nbr_bboxes, tot_nbr_gt_bboxes), dtype=np.float32)

    # Used for indexing appropriately into the columns of
    # bbox_max_ious and gt_max_ious
    ctr_prev = 0
    ctr_curr = 0

    for cls_label in range(1, cfg.NBR_CLASSES):

        # Extract stuff for current class
        bboxes_curr_cls = bboxes[:, (cls_label - 1) * 4:cls_label * 4]
        gt_bboxes_curr_cls = gt_bboxes[gt_bboxes[:, 4] == cls_label][:, 0:4]
        nbr_gt_bboxes = gt_bboxes_curr_cls.shape[0]

        if nbr_gt_bboxes > 0:
            # Increase counter
            ctr_curr += nbr_gt_bboxes

            # Repmat / repeat appropriately for vectorized computations
            gt_bboxes_curr_cls = np.tile(gt_bboxes_curr_cls, [nbr_bboxes, 1])
            bboxes_curr_cls = np.repeat(bboxes_curr_cls,
                                        [nbr_gt_bboxes for _ in range(nbr_bboxes)],
                                        axis=0)

            # Intersection
            ixmin = np.maximum(bboxes_curr_cls[:, 0], gt_bboxes_curr_cls[:, 0])
            iymin = np.maximum(bboxes_curr_cls[:, 1], gt_bboxes_curr_cls[:, 1])
            ixmax = np.minimum(bboxes_curr_cls[:, 2], gt_bboxes_curr_cls[:, 2])
            iymax = np.minimum(bboxes_curr_cls[:, 3], gt_bboxes_curr_cls[:, 3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # Union
            uni = ((bboxes_curr_cls[:, 2] - bboxes_curr_cls[:, 0] + 1.) *
                   (bboxes_curr_cls[:, 3] - bboxes_curr_cls[:, 1] + 1.) +
                   (gt_bboxes_curr_cls[:, 2] - gt_bboxes_curr_cls[:, 0] + 1.) *
                   (gt_bboxes_curr_cls[:, 3] - gt_bboxes_curr_cls[:, 1] + 1.) - inters)

            # IoU
            ious = inters / uni
            ious = np.reshape(ious, [nbr_bboxes, nbr_gt_bboxes])

            # Set everything except row-wise maxes (i.e the values by taking
            # max within each row) to zero, to indicate bbox-gt instance assignments
            # (each bounding box is only allowed to "cover"/be assigned to
            # one instance, as is the case also in mAP evaluation etc.)
            ious[ious - np.max(ious, axis=1)[:, np.newaxis] < 0] = 0

            # Insert into set of all ious (all ious contains ious for all the
            # respective, image-existing categories) and update counter/indexer
            bbox_max_ious[:, ctr_prev:ctr_curr] = ious
            ctr_prev = ctr_curr

    # Also compute maximum coverage for each gt instance
    # (for quicker future computations in e.g. RL reward)
    gt_max_ious = np.max(bbox_max_ious, axis=0)

    # Return
    return bbox_max_ious, gt_max_ious


# This helper function initializes the RL part of the network
# (trainable part, i.e., the non-pretrained part of the network)
def init_rl_variables(sess):
    uninitialized_vars = []
    print("Variables which are now initialized (i.e. they were not loaded!):\n")
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            print(var)
            uninitialized_vars.append(var)
    if len(uninitialized_vars) == 0:
        print("\nNo variables needed initialization (all were instead loaded!)\n")
    else:
        print("\nSuccessfully initialized variables!\n")
    sess.run(tf.variables_initializer(uninitialized_vars))


def _get_rect_coords(fix_h, fix_w, obs_rect_h, obs_rect_w, height, width):
    # The +1 needed due to inclusive in h1, w1, exclusive in h2, w2
    h_start = max(0, fix_h - int(obs_rect_h / 2))
    w_start = max(0, fix_w - int(obs_rect_w / 2))
    h_end = min(height, fix_h + int((obs_rect_h + 1) / 2))
    w_end = min(width, fix_w + int((obs_rect_w + 1) / 2))
    return h_start, w_start, h_end, w_end


# Updates observed voxels (orig)
def update_obs_vol(roi_obs_vol, t, height, width,
                   fix_h, fix_w, obs_rect_h, obs_rect_w):
    # Prior to updating, check which voxels were already observed
    # at time 1,2,..., t-1
    already_obs_rois_idxs_orig = roi_obs_vol > 0
    already_obs_rois_orig = roi_obs_vol[already_obs_rois_idxs_orig]

    # Update RoI volume
    h_start, w_start, h_end, w_end = _get_rect_coords(fix_h, fix_w, obs_rect_h,
                                                      obs_rect_w, height, width)
    # paper says observed voxels' value equal to 1, but here equal to t
    roi_obs_vol[:, h_start:h_end, w_start:w_end, :] = t
    roi_obs_vol[already_obs_rois_idxs_orig] = already_obs_rois_orig

    return roi_obs_vol, h_start, w_start, h_end, w_end


# Check whether to terminate search
def _check_termination(t, done_prob, mode='train'):
    # Check whether to run in training / testing mode
    if mode == 'train':
        random_done = True
    else:
        # False
        random_done = cfg.LRP_HAI_TEST.RANDOM_DONE

    # Basic termination
    if mode == 'test' and cfg.LRP_HAI_TEST.NBR_FIX > 0:
        if t + 1 == cfg.LRP_HAI_TEST.NBR_FIX:
            return True, False
        return False, False
    else:
        if random_done:
            terminate = np.random.uniform() <= done_prob
        else:
            terminate = done_prob > 0.5

    # Used if max number of fixations reached
    if t == cfg.LRP_HAI.MAX_ITER_TRAJ - 1:
        return True, terminate
    else:
        return terminate, True


# Given spatially softmaxed where-to-fix layer, sample such a location to visit
def sample_fix_loc(fix_prob, mode='train'):
    # Check whether to run in training / testing mode
    if mode == 'train':
        random_fix = True
    else:
        random_fix = cfg.LRP_HAI_TEST.RANDOM_FIX

    # Draw uniform random number for location selection
    if random_fix:
        fix_layer_cumulative = np.cumsum(fix_prob)
        u = np.random.rand()
        while u > fix_layer_cumulative[-1]:  # May be round-off errors
            u = np.random.rand()
        first_smaller_than_idx_linear = np.where(u <= fix_layer_cumulative)[0][0]
    else:
        first_smaller_than_idx_linear = np.argmax(fix_prob)

    # Translate back to spatial indexing and form (h,w)-tuple
    fix_loc = np.unravel_index(first_smaller_than_idx_linear, fix_prob.shape)

    # Return (h,w)-tuple
    return fix_loc[0], fix_loc[1], first_smaller_than_idx_linear


# Run LRP-HAI detector on an image blob
def run_LRP_HAI(sess, net, blob, timers, mode, beta, im_idx=None,
                extra_args=None, alpha=True):
    """
    :param alpha:
    :param sess:
    :param net:
    :param blob:
    :param timers:
    :param mode:
    :param beta: from section 5.1.2
    :param im_idx:
    :param extra_args:
    :return:
    """
    # Extract relevant parts from blob (assume 1 img/batch)
    im_blob = blob['data']
    im_info = blob['im_info']
    if mode == 'test':
        im_shape = blob['im_shape_orig']
        im_scale = im_info[2]
    else:
        im_shape = im_info[:2]
        im_scale = 1.0
        gt_boxes = blob['gt_boxes']

    # Run initial LRP-HAI processing (get base feature map etc)
    timers['init'].tic()
    net_conv, rl_in, rl_hid, roi_obs_vol, height, width, \
    fix_rect_h, fix_rect_w = net.get_init_rl(sess, im_blob, im_info)

    # Initialize detection containers
    cls_probs_seqs, bbox_preds_seqs, rois_seqs = [], [], []
    timers['init'].toc()

    # Store and return observation canvas, used for diagnostic / visualization of
    # where LRP-HAI attention has been placed in the trajectory
    obs_canvas = np.zeros((height, width), dtype=np.float32)
    obs_canvas_all = None

    # If training, intitialize certain containers and other things
    if mode == 'train':
        net.reset_pre_traj()
        gt_max_ious = np.zeros(gt_boxes.shape[0], dtype=np.float32)
        rews_total_traj = []
        # rews_done_traj = []

    # Run search trajectory
    timers['fulltraj'].tic()

    # Update Conv-GRU's hidden state first
    timers['action-rl'].tic()
    rl_hid, _, _ = net.action_pass(sess, rl_in, rl_hid, False)
    timers['action-rl'].toc()

    for t in range(cfg.LRP_HAI.MAX_ITER_TRAJ):
        # Update RL state
        if t > 0:

            # Update observation volume (used to keep track of where RoIs have been
            # forwarded for class-specific predictions)
            timers['upd-obs-vol'].tic()
            roi_obs_vol, h_start, w_start, h_end, w_end \
                = update_obs_vol(roi_obs_vol, t, height, width,
                                 fix_h, fix_w, fix_rect_h, fix_rect_w)
            timers['upd-obs-vol'].toc()

            # forward fixate region to RPN
            # crop the fixate region at net_conv
            # fixate_region = tf.image.crop_to_bounding_box(net_conv, h_start, w_start, h_end, w_end)
            fixate_region = net_conv[:, h_start:h_end, w_start:w_end, :]
            rois_fixate = net.pass_to_rpn(sess, fixate_region, h_start, w_start, im_info)

            roi_obs_vec_seq = (roi_obs_vol == t).reshape(-1)

            if np.count_nonzero(roi_obs_vec_seq) > 0:

                # Classify RoIs
                timers['upd-seq'].tic()
                # rois_seq: size(N, 5)
                rois_seq = rois_fixate
                # cls_probs_seq: size(N, num_class), bbox_preds_seq: size(N, num_class*4)
                cls_probs_seq, bbox_preds_seq = net.seq_rois_pass(sess, net_conv,
                                                                  rois_seq, im_info,
                                                                  mode == 'train_det')

                # Add to collection of all detections
                cls_probs_seqs.append(cls_probs_seq)
                bbox_preds_seqs.append(bbox_preds_seq)
                rois_seqs.append(rois_seq)
                rois_seq = rois_seq[:, 1:5] / im_scale

                timers['upd-seq'].toc()
            else:
                rois_seq, cls_probs_seq, bbox_preds_seq = None, None, None

            # Update observation canvas (used in search visualization and
            # to keep track of fraction of spatial area covered by agent)
            obs_canvas[np.squeeze(np.sum(roi_obs_vol == t, 3) > 0)] = 1
            if im_idx is not None:
                if obs_canvas_all is None:
                    obs_canvas_all = np.copy(obs_canvas[:, :, np.newaxis])
                else:
                    curr_canvas = np.zeros_like(obs_canvas)
                    curr_canvas[h_start: h_end, w_start: w_end] = 1
                    obs_canvas_all = np.concatenate([obs_canvas_all,
                                                     curr_canvas[:, :, np.newaxis]], axis=2)

            # Update RL state
            timers['upd-rl'].tic()
            rl_in[:, h_start:h_end, w_start:w_end, :] = -1
            timers['upd-rl'].toc()

            if mode == 'train':

                # Make into final form (want to compute IoU post-bbox regression, so that
                # optimization objective is closer to the final detection task)
                if rois_seq is None:
                    pred_bboxes_fix = None
                else:
                    pred_bboxes_fix = bbox_transform_inv(rois_seq, bbox_preds_seq)
                    pred_bboxes_fix = clip_boxes(pred_bboxes_fix, im_shape)

                # Fixation reward computation
                rew_fix, rew_done, gt_max_ious = net.reward_fixate(pred_bboxes_fix, gt_boxes,
                                                                   gt_max_ious, t, beta)
                rew_total = rew_fix + rew_done
                rews_total_traj.append(rew_total)
                # rews_done_traj.append(rew_done)
        # Action selection (and update of conv-GRU hidden state)
        timers['action-rl'].tic()
        rl_hid, done_prob, fix_prob = net.action_pass(sess, rl_in, rl_hid, mode == 'train')
        timers['action-rl'].toc()

        # Check for termination
        terminate, free_will = _check_termination(t, done_prob[0][0], mode)

        if terminate:
            if mode == 'train':
                rew_done, rew_fix = net.reward_done(fix_prob, t, gt_max_ious, free_will)
                rew_total = rew_fix + rew_done
                rews_total_traj.append(rew_total)
                timers['fulltraj'].toc()
                rews_total_traj = sum(rews_total_traj) / gt_max_ious.shape[0]
                # rew_done = rew_done / gt_max_ious.shape[0]
                frac_gt_covered = float(np.count_nonzero(gt_max_ious >= 0.5)) / len(gt_max_ious)
                frac_gt = np.sum(gt_max_ious) / len(gt_max_ious)
            break

        # If search has not terminated, sample next spatial location to fixate
        fix_h, fix_w, fix_one_hot = sample_fix_loc(fix_prob, mode)
        if mode == 'train':
            net._ep['fix'].append(fix_one_hot)
    timers['fulltraj'].toc()

    # Collect all detections throughout the trajectory
    timers['coll-traj'].tic()
    scores, pred_bboxes, rois, fix_tracker = \
        _collect_detections(rois_seqs, bbox_preds_seqs, cls_probs_seqs, im_shape,
                            im_scale, mode)
    timers['coll-traj'].toc()

    # Save visualization (if desired)
    if im_idx is not None and obs_canvas_all is not None:
        save_visualization(im_blob, im_shape, im_idx, obs_canvas_all, scores,
                           pred_bboxes, fix_tracker, 0, 1)

    # Depending on what mode, return different things
    frac_area = float(np.count_nonzero(obs_canvas)) / np.prod(obs_canvas.shape)
    if mode == 'test':
        if extra_args is not None:
            return scores, pred_bboxes, timers, [t, frac_area]
        else:
            return scores, pred_bboxes, timers, [t, frac_area]
    else:
        return net_conv, rois, gt_boxes, im_info, timers, [rews_total_traj, t, frac_area, frac_gt_covered, frac_gt]


def _collect_detections(rois_seqs, bbox_preds_seqs, cls_probs_seqs, im_shape,
                        im_scale, mode):
    if mode == 'test' or mode == 'train':
        if len(rois_seqs) > 0:
            rois = np.vstack(rois_seqs)
            bbox_pred = np.vstack(bbox_preds_seqs)
            scores = np.vstack(cls_probs_seqs)
            rois = rois / im_scale
            pred_bboxes = bbox_transform_inv(rois[:, 1:5], bbox_pred)
            pred_bboxes = clip_boxes(pred_bboxes, im_shape)

            # Also do sequential collection
            fix_tracker = []
            for i in range(len(rois_seqs)):
                fix_tracker.append(i * np.ones(rois_seqs[i].shape[0], dtype=np.int32))
            fix_tracker = np.hstack(fix_tracker)
        else:
            scores = np.zeros((0, cfg.NBR_CLASSES))
            pred_bboxes = np.zeros((0, 4 * cfg.NBR_CLASSES))
            rois = None
            fix_tracker = None
        return scores, pred_bboxes, rois, fix_tracker


def print_timings(timers):
    start = 1000  # burn-in --> may give fairer average runtimes
    print('init-rl: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['init'].get_avg(), timers['init'].get_avg(start),
           timers['init'].diff))
    print('fultraj: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['fulltraj'].get_avg(), timers['fulltraj'].get_avg(start),
           timers['fulltraj'].diff))
    print('upd-vol: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['upd-obs-vol'].get_avg(), timers['upd-obs-vol'].get_avg(start),
           timers['upd-obs-vol'].diff))
    print('upd-seq: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['upd-seq'].get_avg(), timers['upd-seq'].get_avg(start),
           timers['upd-seq'].diff))
    print('upd-rl:  (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['upd-rl'].get_avg(), timers['upd-rl'].get_avg(start),
           timers['upd-rl'].diff))
    print('action:  (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['action-rl'].get_avg(), timers['action-rl'].get_avg(start),
           timers['action-rl'].diff))
    print('col-tra: (tot, post1k, curr) (%.4f, %.4f, %.4f)' % \
          (timers['coll-traj'].get_avg(), timers['coll-traj'].get_avg(start),
           timers['coll-traj'].diff))


def produce_det_bboxes(im, scores, det_bboxes, fix_tracker, thresh_post=0.80,
                       thresh_pre=0.0):
    """
  Based on the forward pass in a detector, extract final detection
  bounding boxes with class names and class probablity scores
  """
    class_names = cfg.CLASS_NAMES[0]
    # class_names = ['bg', ' aero', 'bike', 'bird', 'boat', 'bottle', 'bus', 'car',
    #                'cat', 'chair', 'cow', 'table', 'dog', 'horse', 'moto', 'person',
    #                'plant', 'sheep', 'sofa', 'train', 'tv']
    # height, width = im.shape[:2]
    # colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0],
    #           [0.5804, 0, 0.82745], [1, 0, 1], [0, 1, 1],
    #           [0, 1, 0.498]]
    # colors = [[0, 1, 0.498], [1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0],
    #           [0.5804, 0, 0.82745], [1, 0, 1], [0, 1, 1]]

    # nbr_colors = len(colors)
    # col_idx = 0
    names_and_coords = []
    cls_dets_all = []
    for j in range(1, cfg.NBR_CLASSES):
        inds = np.where(scores[:, j] > thresh_pre)[0]
        cls_scores = scores[inds, j]
        cls_bboxes = det_bboxes[inds, j * 4:(j + 1) * 4]
        curr_fix_tracker = fix_tracker[inds]
        cls_dets = np.hstack((cls_bboxes, cls_scores[:, np.newaxis]))
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        curr_fix_tracker = curr_fix_tracker[keep]
        keep = cls_scores[keep] > thresh_post
        cls_dets = cls_dets[keep]
        curr_fix_tracker = curr_fix_tracker[keep]
        name = class_names[j]
        colors = (np.random.random((1, 3)) * 0.7 + 0.2).tolist()[0]
        for jj in range(cls_dets.shape[0]):
            crop = np.squeeze(cls_dets[jj, :])
            cls_dets_all.append(crop[:4])
            coords = [crop[0], crop[1]]
            names_and_coords.append({'coords': coords,
                                     'score': round(crop[4], 2),
                                     'class_name': name,
                                     'color': colors,
                                     'fix': curr_fix_tracker[jj]})
        # if (cls_dets.shape[0]) > 0:
        #     col_idx += 1
        #     col_idx %= nbr_colors
    return cls_dets_all, names_and_coords


def save_visualization(im_blob, im_shape, im_idx, obs_canvas, cls_probs,
                       det_bboxes, fix_tracker, show_all_steps=False,
                       show_text=True):
    # Make sure image in right range
    im = im_blob[0, :, :, :]
    im -= np.min(im)
    im /= np.max(im)
    im = resize(im, (im_shape[0], im_shape[1]), order=1, mode='reflect')

    # BGR --> RGB
    im = im[..., ::-1]

    # Make sure obs_canvas has same size as im
    obs_canvas = resize(obs_canvas, (im.shape[0], im.shape[1]), order=1,
                        mode='reflect')

    # Produce final detections post-NMS
    cls_dets, names_and_coords = produce_det_bboxes(im, cls_probs,
                                                    det_bboxes, fix_tracker)

    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    # Show image
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    # Potentially we may want to show-step-by-step
    if show_all_steps:
        save_ctr = 0
        im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg'
        plt.savefig(im_name)

    # Draw all fixation rectangles
    for i in range(obs_canvas.shape[2]):

        # Extract current stuff
        if np.count_nonzero(obs_canvas[:, :, i]) == 0:
            continue
        nonzeros = np.nonzero(obs_canvas[:, :, i])
        start_x = nonzeros[1][0]
        start_y = nonzeros[0][0]
        end_x = nonzeros[1][-1]
        end_y = nonzeros[0][-1]

        # Show fixation number
        if show_text:
            ax.text(start_x, start_y, "glimpse " + str(i + 1), color='black', weight='bold',
                    fontsize=8,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=dict(facecolor='white', edgecolor='white', pad=-0.1))

        # Show fixation rectangle
        rect = patches.Rectangle((start_x, start_y), end_x - start_x,
                                 end_y - start_y,
                                 linewidth=4, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

        # Potentially we may want to show-step-by-step
        if show_all_steps:
            save_ctr += 1
            im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg'
            plt.savefig(im_name)

        # Draw all detection boxes
        for j in range(len(names_and_coords)):

            # Extract current stuff
            fix = names_and_coords[j]['fix']
            if fix != i:
                continue
            coords = names_and_coords[j]['coords']
            score = names_and_coords[j]['score']
            name = names_and_coords[j]['class_name']
            color = names_and_coords[j]['color']
            cls_det = cls_dets[j]

            # Show object category + confidence
            if show_text:
                ax.text(coords[0], coords[1], name + " " + str(score),
                        weight='bold', color='black', fontsize=8,
                        horizontalalignment='center', verticalalignment='center',
                        bbox=dict(facecolor=color, edgecolor=color, pad=-0.1))

            # Show detection bounding boxes
            rect = patches.Rectangle((cls_det[0], cls_det[1]), cls_det[2] - cls_det[0],
                                     cls_det[3] - cls_det[1],
                                     linewidth=4, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        # Potentially we may want to show-step-by-step
        if show_all_steps:
            save_ctr += 1
            im_name = 'im' + str(im_idx + 1) + '_' + str(save_ctr) + '.jpg'
            plt.savefig(im_name)

    # Final save / close of figure
    if ~show_all_steps:
        # fig = plt.gcf()
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        im_name = 'im' + str(im_idx + 1) + '.jpg'
        # fig.savefig(im_name, transparent=True, dpi=300, bbox_inches='tight', pad_inches=0.0)
        plt.savefig(im_name, transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # Display success message
    print("Saved image " + im_name + "!\n")
