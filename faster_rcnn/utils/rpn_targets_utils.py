from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor


def generate_rpn_targets(
    all_anchors: Tensor,
    gt_bbox: Tensor,
    im_shape: Tuple[int, int],
    margin: int = 100,
    clobber_positive: bool = False,
    neg_iou_thresh: float = 0.3,
    pos_iou_thresh: float = 0.7,
    pos_anchors_perc: float = 0.5,
    anchor_batch: int = 256,
) -> Tuple[Tensor, Tensor, Tensor]:
    """[summary]

    Args:
        all_anchors (Tensor): [description]
        gt_bbox (Tensor): [description]
        im_shape (Tuple[int, int]): [description]

    Returns:
        Tuple[Tensor, Tensor, Tensor]: [description]
    """

    H, W = im_shape
    gt_bbox = gt_bbox[:, :-1]

    # filter anchors; only achors inside the image margin should be kept, ignore
    # anchors outside image
    x_min, y_min, x_max, y_max = tf.unstack(all_anchors, 4, 1)
    greater_than_or_zero = tf.logical_and((x_min >= 0 - margin), (y_min >= 0 - margin))
    less_than_image_w_and_h = tf.logical_and((x_max < W + margin), (y_max < H + margin))
    anchor_filter_ = tf.logical_and(greater_than_or_zero, less_than_image_w_and_h)
    anchor_filter_ind = tf.where(anchor_filter_)
    anchors = tf.boolean_mask(all_anchors, anchor_filter_)

    # array with labels for all anchors, -1: ingnored, 0: negative, 1:positive
    n_anchors = anchors.shape[0]
    labels = tf.fill([n_anchors], -1)

    # intersection over union between anchors and ground truth boxes
    overlaps = bbox_overlap_tf(anchors, gt_bbox)

    # array with closest ground truth value for each anchor
    anchor_max_gt_overlaps = tf.reduce_max(overlaps, axis=1)

    if not clobber_positive:
        # assign background labels first so that positive labels can
        # clobber them latter
        neg_anchor_filter = tf.less(anchor_max_gt_overlaps, neg_iou_thresh)

        # neg anchors set to zero
        labels = tf.where(neg_anchor_filter, tf.zeros_like(labels), labels)

    # array with max anchor iou value for each ground truth box
    gt_max_achor_overlaps = tf.reduce_max(overlaps, axis=0)

    # obtain all anchors for each ground truth with the gt_max_achor_overlaps
    # and anchors associated with the max iou for each ground truth are considered
    # positive
    filter_ = tf.equal(overlaps, gt_max_achor_overlaps)
    filter_ = tf.reduce_sum(tf.cast(filter_, tf.int32), axis=1)
    filter_ = tf.cast(filter_, tf.bool)
    labels = tf.where(filter_, tf.ones_like(labels), labels)

    # anchors with iou greater than the positive threshold are considered positive
    filter_ = anchor_max_gt_overlaps >= pos_iou_thresh
    labels = tf.where(filter_, tf.ones_like(labels), labels)

    if clobber_positive:
        # assign background labels last so that they can clobber positive labels
        neg_anchor_filter = tf.less(anchor_max_gt_overlaps, neg_iou_thresh)

        # neg anchors set to zero
        labels = tf.where(neg_anchor_filter, tf.zeros_like(labels), labels)

    # we remove some positive anchors if positive anchors are too much
    # pos anchors should not be greater than POS_ANCHORS_PERC or ANCHOR_BATCH
    num_pos_anchors_thresh = pos_anchors_perc * anchor_batch
    positive_anchor_ind = tf.where(labels == 1)
    num_pos_anchors = len(positive_anchor_ind)
    if num_pos_anchors >= num_pos_anchors_thresh:
        # reduce number of positive anchors
        num_indices_to_disable = tf.cast(
            num_pos_anchors - num_pos_anchors_thresh, tf.int32
        )
        positive_anchor_ind = tf.random.shuffle(positive_anchor_ind)
        positive_anchor_ind_to_disable = positive_anchor_ind[:num_indices_to_disable]

        positive_anchor_ind_to_disable = tf.expand_dims(
            positive_anchor_ind_to_disable, axis=-1
        )
        positive_anchor_ind_to_disable = tf.SparseTensor(
            tf.cast(positive_anchor_ind_to_disable, tf.int64),
            [1] * len(positive_anchor_ind_to_disable),
            [len(labels)],
        )
        positive_anchor_ind_to_disable = tf.sparse.to_dense(
            positive_anchor_ind_to_disable, default_value=False, validate_indices=False
        )
        positive_anchor_ind_to_disable_filter = tf.cast(
            positive_anchor_ind_to_disable, tf.bool
        )
        labels = tf.where(
            positive_anchor_ind_to_disable_filter, tf.fill(len(labels), -1), labels
        )
        num_pos_anchors = tf.reduce_sum(tf.cast(labels == 1, tf.int32))

    num_neg_anchors_thresh = anchor_batch - num_pos_anchors
    negative_anchor_ind = tf.where(labels == 0)
    num_neg_anchors = len(negative_anchor_ind)
    if num_neg_anchors >= num_neg_anchors_thresh:
        # reduce number of negative anchors
        num_indices_to_disable = tf.cast(
            num_neg_anchors - num_neg_anchors_thresh, tf.int32
        )
        negative_anchor_ind = tf.random.shuffle(negative_anchor_ind)
        negative_anchor_ind_to_disable = negative_anchor_ind[:num_indices_to_disable][
            :, 0
        ]

        negative_anchor_ind_to_disable = tf.expand_dims(
            negative_anchor_ind_to_disable, axis=-1
        )
        negative_anchor_ind_to_disable = tf.SparseTensor(
            tf.cast(negative_anchor_ind_to_disable, tf.int64),
            [1] * len(negative_anchor_ind_to_disable),
            [len(labels)],
        )
        negative_anchor_ind_to_disable = tf.sparse.to_dense(
            negative_anchor_ind_to_disable, default_value=False, validate_indices=False
        )
        negative_anchor_ind_to_disable_filter = tf.cast(
            negative_anchor_ind_to_disable, tf.bool
        )
        labels = tf.where(
            negative_anchor_ind_to_disable_filter, tf.fill(len(labels), -1), labels
        )

    # obtain anchor targets
    # which ground truth box is closest to an anchor
    anchor_gt_ind = tf.argmax(overlaps, axis=1)
    anchor_gt_boxes = tf.gather(gt_bbox, anchor_gt_ind)

    # obtain the targets using the transform
    bbox_targets = encode(anchors, anchor_gt_boxes)

    # ignore targets for non positive anchors
    filter_ = tf.reshape(labels > 0, [-1, 1])
    bbox_targets = tf.where(filter_, bbox_targets, tf.zeros_like(bbox_targets))

    # for shape compatibility
    bbox_targets = tf.scatter_nd(anchor_filter_ind, bbox_targets, all_anchors.shape)
    labels = tf.scatter_nd(anchor_filter_ind, labels, [all_anchors.shape[0]])
    labels = tf.where(anchor_filter_, labels, tf.fill(labels.shape, -1))
    max_overlap = tf.scatter_nd(
        anchor_filter_ind, anchor_max_gt_overlaps, [all_anchors.shape[0]]
    )

    assert bbox_targets.shape == all_anchors.shape

    return bbox_targets, labels, max_overlap
