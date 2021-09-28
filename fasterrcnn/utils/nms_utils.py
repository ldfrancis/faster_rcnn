from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor

from .bbox_utils import swap_xy
from .proposals_utils import filter_proposals


  
def apply_nms(
    bboxes: Tensor, scores: Tensor, nms_threshold: Tensor, top_n: Tensor
) -> Tuple[Tensor, Tensor]:
    """Applies non-max-suppression to set of bounding boxes using their associated
    scores and nms_threshold

    Args:
        bboxes (Tensor): The bounding boxes to apply non-max-suppression on. 2-D
        float32 Tensor of shape (num_bboxes, 4)
        scores (Tensor): The scores (objectnes score or classification score) of
        each bounding box. 1-D float32 Tensor of shape (num_bboxes,)
        nms_threshold (Tensor): The threshold (for iou) above which bboxes with
        lower scores would be suppressed/removed. 0-D float32 Tensor, Scalar.
        top_n (Tensor): The number of bounding boxes that should be kept or should
        remain after nms. 0-D int32 Tensor, Scalar.

    Returns:
        Tuple[Tensor, Tensor]: The bounding boxes resulting from nms and their scores.
        They are ordered by the scores
    """
    selected_indces = tf.image.non_max_suppression(
        swap_xy(bboxes), scores, top_n, nms_threshold
    )
    bboxes = tf.gather(bboxes, selected_indces, axis=0)
    scores = tf.gather(scores, selected_indces, axis=0)

    return bboxes, scores


def per_class_nms(
    bboxes: Tensor,
    class_: Tensor,
    scores: Tensor,
    im_size: Tensor,
    score_thresh: Tensor = tf.constant(0.7),
    nms_threshold: Tensor = tf.constant(0.7),
    top_n: Tensor = tf.constant(19),
) -> Tuple[Tensor, Tensor]:
    """Non-max-suppression for bounding boxes of each object class in a dataset

    Args:
        bboxes (Tensor): The bounding boxes. 2-D float32 Tensor of shape (num_bboxes, 4)
        class_ (Tensor): The associated class for each bounding box. 1-D int32 Tensor of
        shape (num_bboxes,)
        scores (Tensor): The classification score for each bounding box. 1-D float32
        Tensor of shape (num_bboxes,)
        im_size (Tensor): The dimensions of the image for which objects are to be
        detected. 1-D int32 Tensor of shape (2,)
        score_thresh (Tensor, optional): The classification score threshold. 0-D float32
        Tensor, Scalar. Defaults to tf.constant(0.7).
        nms_threshold (Tensor, optional): The threshold for nms. 0-D float32 Tensor,
        Scalar. Defaults to tf.constant(0.7).
        top_n (Tensor, optional): The possible number of bounding boxes to keep after
        nms. 0-D int32 Tensor, Scalar. Defaults to tf.constant(19).

    Returns:
        Tuple[Tensor, Tensor]: The bounding boxes and respective scores
    """
    uniques_clases, _ = tf.unique(class_)

    temp_bboxes = []
    temp_scores = []

    for cls in uniques_clases:
        cls_bboxes = tf.boolean_mask(bboxes, class_ == cls)
        cls_scores = tf.boolean_mask(scores, class_ == cls)
        cls_bboxes, cls_scores = filter_proposals(
            cls_bboxes, cls_scores, im_size, score_thresh, True
        )
        cls_bboxes, cls_scores = apply_nms(cls_bboxes, cls_scores, nms_threshold, top_n)
        cls_bboxes = tf.concat(
            [cls_bboxes, tf.fill([len(cls_bboxes), 1], tf.cast(cls, tf.float32))],
            axis=-1,
        )
        temp_bboxes += [cls_bboxes]
        temp_scores += [cls_scores]

    bboxes = tf.concat(temp_bboxes, axis=0)
    scores = tf.concat(temp_scores, axis=0)

    return bboxes, scores
