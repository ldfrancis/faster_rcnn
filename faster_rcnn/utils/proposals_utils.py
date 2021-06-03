from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor


# proposals_util
def filter_proposals(
    proposals, scores, im_size, score_thresh: float = 0.7, inference=False
) -> Tuple[Tensor, Tensor]:
    """Filters non-positive area proposals.

    Arguments:
        proposals: Tensor of shape (num_proposals, 4), holding the
            coordinates of the proposals' bounding boxes.
        scores: Tensor of shape (num_proposals,), holding the
            scores associated to each bounding box.

    Returns:
        (`proposals`, `scores`), but with non-positive area proposals removed.
    """

    x1, y1, x2, y2 = tf.unstack(proposals, 4, axis=1)

    if inference:
        box_area = (x2 - x1) * (y2 - y1)
        positive_area_mask = tf.greater(box_area, 0)
        filter_ = positive_area_mask

        # positive x, y
        positive_x = tf.logical_and(x1 > 0, x2 > 0)
        positive_y = tf.logical_and(y1 > 0, y2 > 0)
        positive_x_y_mask = tf.logical_and(positive_x, positive_y)
        filter_ = tf.logical_and(filter_, positive_x_y_mask)

        # x, y within bounds
        H, W = im_size
        within_bounds_x = tf.logical_and(x1 < W, x2 < W)
        within_bounds_y = tf.logical_and(y1 < H, y2 < H)
        within_bounds_mask = tf.logical_and(within_bounds_x, within_bounds_y)
        filter_ = tf.logical_and(filter_, within_bounds_mask)

        # positive area
        box_area = (x2 - x1) * (y2 - y1)
        positive_area_mask = tf.greater(box_area, 0)
        filter_ = positive_area_mask

        # positive box score
        score_thresh = scores > score_thresh
        filter_ = tf.logical_and(filter_, score_thresh)
    else:
        H, W = im_size
        x1 = tf.clip_by_value(x1, 0, W - 1)
        y1 = tf.clip_by_value(y1, 0, H - 1)
        x2 = tf.clip_by_value(x2, 0, W - 1)
        y2 = tf.clip_by_value(y2, 0, H - 1)
        proposals = tf.stack([x1, y1, x2, y2], axis=1)

        box_area = (x2 - x1) * (y2 - y1)
        positive_area_mask = tf.greater(box_area, 0)
        filter_ = positive_area_mask

    # filter boxes
    boxes = tf.boolean_mask(proposals, filter_, axis=0)
    scores = tf.boolean_mask(scores, filter_)

    return boxes, scores
