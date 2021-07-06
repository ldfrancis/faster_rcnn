# This implementation was inspired by tryolab's luminoth
# https://github.com/tryolabs/luminoth

import pdb
from typing import Tuple

import tensorflow as tf
from fasterrcnn.utils.bbox_utils import bbox_overlap, encode
from tensorflow import Tensor


@tf.function
def generate_rpn_targets(
    anchors: Tensor,
    gt_bboxes: Tensor,
    im_size: Tensor,
    margin: Tensor,
    clobber_positive: Tensor = tf.constant(True),
    neg_iou_thresh: Tensor = tf.constant(0.3),
    pos_iou_thresh: Tensor = tf.constant(0.7),
    pos_anchors_perc: Tensor = tf.constant(0.5),
    anchor_batch: Tensor = tf.constant(2),
) -> Tuple[Tensor, Tensor]:
    """Generates the targets for anchors to serve as expected predictions from the RPN
    in faster-rcnn.

    Args:
        anchors (Tensor): The anchors. 2-D float32 Tensor of shape [num_anchors,4]
        gt_bboxes (Tensor): The ground truth bounding boxes. 2-D float32 Tensor of shape
        [num_ground_truth_boxes, 5]
        im_size (Tensor): Contains width and height of the image. 1-D int32 Tensor of
        shape [2,]
        margin (Tensor): Margin used to determine anchors with the image boundary. 0-D
        int32 Tensor, Scalar
        clobber_positive (Tensor, optional): Used to decide whether to assign positive
        anchor labels first or not. This decision determines whether negative labels
        would clobber/overwrite positive ones. 0-D boolean Tensor, Scalar.
        Defaults to tf.constant(True).
        neg_iou_thresh (Tensor, optional): Threshold used to determine if an anchor is
        positive. 0-D float32 Tensor, Scalar. Defaults to tf.constant(0.3).
        pos_iou_thresh (Tensor, optional): Threshold ussed to determine if an anchor is
        positive. 0-D float32 Tensor, Scalar. Defaults to tf.constant(0.7).
        pos_anchors_perc (Tensor, optional): Percentage/fraction of positive anchors to
        be found in the batch of anchors. 0-D float32 Tensor, Scalar. Defaults to
        tf.constant(0.5).
        anchor_batch (Tensor, optional): The number of anchors to consider in a batch.
        0-D int32 Tensor, Scalar. Defaults to tf.constant(2).

    Returns:
        Tuple[Tensor, Tensor]: The targets and labels. 2-D float32 Tensor and 1-D Tensor
    """

    n_anchors, _ = anchors.shape
    height = tf.cast(im_size[0], tf.float32)
    width = tf.cast(im_size[1], tf.float32)
    margin = tf.cast(margin, tf.float32)
    anchor_batch = tf.cast(anchor_batch, tf.float32)

    labels = tf.fill([n_anchors], -1)
    bboxes = gt_bboxes[:, :4]

    # ignore anchors outside margin
    x1, y1, x2, y2 = tf.unstack(anchors, 4, 1)
    filter = tf.logical_and(tf.greater(x1, -margin), tf.greater(y1, -margin))
    filter = tf.logical_and(
        filter,
        tf.logical_and(tf.less(x2, width + margin), tf.less(y2, height + margin)),
    )
    valid_anchor_indices = tf.where(filter)

    anchors = tf.boolean_mask(anchors, filter)
    labels = tf.boolean_mask(labels, filter)

    overlaps = bbox_overlap(anchors, bboxes)
    anchor_max_overlaps = tf.reduce_max(overlaps, axis=1)
    gt_max_overlaps = tf.reduce_max(overlaps, axis=0)

    # should background/negative anchor labels be assigned first?
    if not clobber_positive:
        neg_label_condition = tf.less_equal(anchor_max_overlaps, neg_iou_thresh)
        labels = tf.where(neg_label_condition, tf.zeros_like(labels), labels)

    # assign positive anchors
    # anchors with max overlaps with any ground truth should be labeled positive
    pos_label_condition = tf.reduce_any(tf.equal(overlaps, gt_max_overlaps), axis=1)
    pos_label_condition = tf.logical_or(
        pos_label_condition, tf.greater_equal(anchor_max_overlaps, pos_iou_thresh)
    )
    labels = tf.where(pos_label_condition, tf.ones_like(labels), labels)

    # should background/negative labels be assigned last?
    if clobber_positive:
        neg_label_condition = tf.less(anchor_max_overlaps, neg_iou_thresh)
        labels = tf.where(neg_label_condition, tf.zeros_like(labels), labels)

    # if the positive anchors are too much (i.e) more than the allowed maximum as
    # suggested by achor_batch and pos_anchor_perc, we disable some so that the
    # maximum is not exceeded.
    max_num_pos_anchors = tf.cast(pos_anchors_perc * anchor_batch, tf.int32)
    num_pos_anchors = tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.int32))
    if num_pos_anchors > max_num_pos_anchors:
        num_pos_anchors_to_disable = num_pos_anchors - max_num_pos_anchors
        pos_anchors_indices = tf.where(labels == 1)
        indices_to_disable = tf.random.shuffle(pos_anchors_indices)[
            :num_pos_anchors_to_disable
        ]
        labels = tf.tensor_scatter_nd_update(
            labels, indices_to_disable, tf.fill([num_pos_anchors_to_disable], -1)
        )
        num_pos_anchors = tf.reduce_sum(tf.cast(tf.equal(labels, 1), tf.int32))

    # same goes for the negative anchors. They must sum up to the anchor_batch
    max_num_neg_anchors = tf.cast(anchor_batch, tf.int32) - num_pos_anchors
    num_neg_anchors = tf.reduce_sum(tf.cast(tf.equal(labels, 0), tf.int32))
    if num_neg_anchors > max_num_neg_anchors:
        num_neg_anchors_to_disable = num_neg_anchors - max_num_neg_anchors
        neg_anchors_indices = tf.where(labels == 0)
        indices_to_disable = tf.random.shuffle(neg_anchors_indices)[
            :num_neg_anchors_to_disable
        ]
        labels = tf.tensor_scatter_nd_update(
            labels, indices_to_disable, tf.fill([num_neg_anchors_to_disable], -1)
        )

    # obtain the ground truth box for each anchor box
    gt_bboxes_indices = tf.argmax(overlaps, axis=1)
    gt_bboxes_for_anchors = tf.gather(bboxes, gt_bboxes_indices)

    targets = encode(anchors, gt_bboxes_for_anchors)

    targets = tf.scatter_nd(
        valid_anchor_indices,
        targets,
        tf.constant([n_anchors, 4], dtype=valid_anchor_indices.dtype),
    )
    labels = tf.tensor_scatter_nd_update(
        tf.fill([n_anchors], -1),
        valid_anchor_indices,
        labels,
    )

    return targets, labels
