from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor


@tf.function
def filter_proposals(
    proposals: Tensor,
    scores: Tensor,
    im_size: Tensor,
    score_thresh: Tensor = 0.7,
    inference: Tensor = False,
) -> Tuple[Tensor, Tensor]:
    """Filters the proposed bounding boxes using the area and scores. Only boxes with
    positive area and score above the threshold are kept. At train time we only filter
    using the positive area, but at inference time, we use both positive area and score
    threshold

    Args:
        proposals (Tensor): The proposed bounding boxes. 2-D float32 Tensor of shape
         (num_proposals, 4)
        scores (Tensor): The scores of the proposed bounding boxes (This could be
         objectness or classification scores). 1-D float32 Tensor of shape
         (num_proposals)
        im_size (Tensor): The dimension of the image for which bounding boxes are
         proposed. 1-D int32 Tensor of shape (2,)
        score_thresh (Tensor, optional): The score threshold. 0-D float32 Tensor, Scalar
         Defaults to 0.7.
        inference (Tensor, optional): Train or Inference. 0-D bool Tensor, Scalae.
         Defaults to False.

    Returns:
        (proposals, scores) Tuple[Tensor, Tensor]: The filters proposals and their
        respective scores.
    """

    x1, y1, x2, y2 = tf.unstack(proposals, 4, axis=1)

    H, W = im_size[0], im_size[1]

    x1 = tf.clip_by_value(x1, tf.cast(0, tf.float32), tf.cast(W - 1, tf.float32))
    y1 = tf.clip_by_value(y1, tf.cast(0, tf.float32), tf.cast(H - 1, tf.float32))
    x2 = tf.clip_by_value(x2, tf.cast(0, tf.float32), tf.cast(W - 1, tf.float32))
    y2 = tf.clip_by_value(y2, tf.cast(0, tf.float32), tf.cast(H - 1, tf.float32))

    proposals = tf.stack([x1, y1, x2, y2], axis=1)

    box_area = (x2 - x1) * (y2 - y1)
    positive_area_mask = tf.greater(box_area, 0)
    filter_ = positive_area_mask

    if inference:
        # positive box score
        score_thresh = scores > score_thresh
        filter_ = tf.logical_and(filter_, score_thresh)

    # filter proposals
    proposals = tf.boolean_mask(proposals, filter_)
    scores = tf.boolean_mask(scores, filter_)

    return proposals, scores
