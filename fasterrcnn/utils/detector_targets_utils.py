from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor

from .bbox_utils import bbox_overlap, encode


@tf.function
def generate_detector_targets(
    proposals: Tensor,
    gt_bboxes: Tensor,
    bg_low: Tensor,
    bg_high: Tensor,
    fg_low: Tensor,
    pos_prop_perc: Tensor,
    prop_batch: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Generates the targets for the proposed bounding boxes for the detector network
    of faster rcnn

    Args:
        proposals (Tensor): The proposed bounding boxes for image regions from the
        region proposal network. 2-D float32 Tensor of shape (num_proposals, 4)
        gt_bboxes (Tensor): The ground truth bounding boxes. 2-D float32 Tensor of
        shape (num_ground_truth_boxes, 5)
        bg_low (Tensor): The iou value above which a proposed bounding box would be
        considered a background. 0-D float32 Tensor, Scalar
        bg_high (Tensor): The iou value below which a proposed bounding box should be
        considered a backgroud. 0-D float32 Tensor, Scalar
        fg_low (Tensor): The iou above which a proposed bounding box would be considered
        a foreground. 0-D float32 Tensor, Scalar
        pos_prop_perc (Tensor): The percentage/fraction of foreground proposals to be
        included in the batch of proposed boxes. 0-D float32 Tensor, Scalar
        prop_batch (Tensor): The number of proposed boxes to be considered. 0-D int32
        Tensor, Scalar

    Returns:
        Tuple[Tensor, Tensor]: The targets and labels for the proposed bounding boxes
    """
    bboxes = gt_bboxes[:, :4]

    # labels for proposals. 0 -> background. (1-num_classes+1) -> foreground/objects.
    # -1 -> ignored. initially all proposals are ignored and labels are later assigned
    # based on intersection with ground truth boxes
    labels = tf.cast(tf.ones_like(proposals)[:, 0] * -1, tf.int32)

    overlaps = bbox_overlap(proposals, bboxes)

    proposal_max_overlaps = tf.reduce_max(overlaps, axis=1)
    gt_max_overlaps = tf.reduce_max(overlaps, axis=0)

    # assign backgroud labels, 0
    labels = tf.where(
        tf.logical_and(
            tf.greater_equal(proposal_max_overlaps, bg_low),
            tf.less(proposal_max_overlaps, bg_high),
        ),
        tf.zeros_like(labels),
        labels,
    )

    # assign foreground labels
    gt_for_proposals = tf.argmax(overlaps, axis=1)

    gt_label_for_proposals = (
        tf.gather(tf.cast(gt_bboxes[:, 4], tf.int32), gt_for_proposals) + 1
    )
    labels = tf.where(
        tf.logical_or(
            tf.greater_equal(proposal_max_overlaps, fg_low),
            tf.reduce_any(overlaps == gt_max_overlaps, axis=1),
        ),
        gt_label_for_proposals,
        labels,
    )

    # ensure that the number of foreground proposals is not above the threshold as
    # determined by pos_prop_perc and prop_batch
    max_num_fg = tf.cast(pos_prop_perc * tf.cast(prop_batch, tf.float32), tf.int32)
    num_fg = tf.reduce_sum(tf.cast(labels > 0, tf.int32))
    if num_fg > max_num_fg:
        num_fg_to_disable = num_fg - max_num_fg
        fg_indices = tf.random.shuffle(tf.where(labels > 0))
        fg_indices_to_disable = fg_indices[:num_fg_to_disable]
        labels = tf.tensor_scatter_nd_update(
            labels, fg_indices_to_disable, tf.fill([num_fg_to_disable], -1)
        )
        num_fg = tf.reduce_sum(tf.cast(labels > 0, tf.int32))
        tf.assert_equal(num_fg, max_num_fg)

    # also ensure the number of background proposal, when added to the foreground,
    # add up to prop_batch
    max_num_bg = tf.cast(prop_batch, tf.int32) - num_fg
    num_bg = tf.reduce_sum(tf.cast(labels == 0, tf.int32))
    if num_bg > max_num_bg:
        num_bg_to_disable = num_bg - max_num_bg
        bg_indices = tf.random.shuffle(tf.where(labels == 0))
        bg_indices_to_disable = bg_indices[:num_bg_to_disable]
        labels = tf.tensor_scatter_nd_update(
            labels, bg_indices_to_disable, tf.fill([num_bg_to_disable], -1)
        )
        num_bg = tf.reduce_sum(tf.cast(labels == 0, tf.int32))
        tf.assert_equal(num_bg, max_num_bg)

    # encode proposals
    gt_bbox = tf.gather_nd(bboxes, tf.expand_dims(gt_for_proposals, 1))
    targets = encode(proposals, gt_bbox)

    return targets, labels
