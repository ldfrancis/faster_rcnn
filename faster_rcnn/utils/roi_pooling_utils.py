import tensorflow as tf


# roipooling_util
def normalize_bboxes(proposals, im_shape):
    """
    Gets normalized coordinates for RoIs (between 0 and 1 for cropping)
    in TensorFlow's order (y1, x1, y2, x2).

    Arguments:
        roi_proposals: A Tensor with the bounding boxes of shape
            (total_proposals, 4), where the values for each proposal are
            (x_min, y_min, x_max, y_max).
        im_shape: A Tensor with the shape of the image (height, width).

    Returns:
        bboxes: A Tensor with normalized bounding boxes in TensorFlow's
            format order. Its should is (total_proposals, 4).
    """

    height, width = im_shape
    x1, y1, x2, y2 = tf.unstack(proposals, axis=-1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    x2 = tf.clip_by_value(x2, 0, width - 1)
    y2 = tf.clip_by_value(y2, 0, height - 1)
    y1, y2 = y1 / (height - 1), y2 / (height - 1)
    x1, x2 = x1 / (width - 1), x2 / (width - 1)

    bboxes = tf.stack([y1, x1, y2, x2], axis=1)
    bboxes = tf.clip_by_value(bboxes, 0, 1)

    return bboxes


def roi_pooling(feature_map, proposals, im_shape, pool_size=7):
    """Perform RoI pooling.

    This is a simplified method than what's done in the paper that obtains
    similar results. We crop the proposal over the feature map and resize it
    bilinearly.

    This function first resizes to *double* of `pool_size` (i.e. gets
    regions of (pool_size * 2, pool_size * 2)) and then uses max pooling to
    get the final `(pool_size, pool_size)` regions.

    Arguments:
        feature_map: Tensor of shape (1, W, H, C), with WxH the spatial
            shape of the feature map and C the number of channels (1024
            in this case).
        proposals: Tensor of shape (total_proposals, 4), holding the proposals
            to perform RoI pooling on.
        im_shape: A Tensor with the shape of the image (height, width).
        pool_size (int): Final width/height of the pooled region.

    Returns:
        Pooled feature map, with shape `(num_proposals, pool_size, pool_size,
        feature_map_channels)`.
    """

    crop_size = pool_size * 2
    croped = tf.image.crop_and_resize(
        feature_map,
        normalize_bboxes(proposals, im_shape),
        [0] * len(proposals),
        [crop_size] * 2,
    )

    rois = tf.keras.layers.MaxPool2D()(croped)

    return rois
