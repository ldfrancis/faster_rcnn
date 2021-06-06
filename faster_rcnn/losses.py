import tensorflow as tf


# loss_util
def smooth_l1_loss(bbox_prediction, bbox_target):
    """
    Return Smooth L1 Loss for bounding box prediction.
    Args:
        bbox_prediction: shape (1, H, W, num_anchors * 4)
        bbox_target:     shape (1, H, W, num_anchors * 4)
    Smooth L1 loss is defined as:
    0.5 * x^2                  if |x| < d
    abs(x) - 0.5               if |x| >= d
    Where d = 1 and x = prediction - target
    """
    diff = bbox_prediction - bbox_target
    abs_diff = tf.abs(diff)
    abs_diff_filter = abs_diff < 1.0
    bbox_loss = tf.reduce_sum(
        tf.where(abs_diff_filter, 0.5 * tf.square(abs_diff), abs_diff - 0.5), [1]
    )
    return bbox_loss
