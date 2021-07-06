""" Adopted from the tryolabs object detection workshop on faster rcnn 
https://github.com/tryolabs/object-detection-workshop
"""

import tensorflow as tf
from tensorflow import Tensor


@tf.function
def normalize_bboxes(proposals: Tensor, width: Tensor, height: Tensor) -> Tensor:
    """Normalizes the proposal bounding box cordinates to be in [0,1] using the width
    and height of the image. The normalized bounding box cordinates are ordered as
    y1,x1,y2,x2 to conform to tensorflow's convention. These are used to obtain ROI
    features from the feature map during ROI POOLING step

    Args:
        proposals (Tensor): The proposed bounding boxes from the RPN. 2-D float32 Tensor
         of shape (num_proposals, 4)
        width (Tensor): The widht of the image for which objects are to be detected. 0-D
         int32 Tensor, Scalar
        height (Tensor): The height of the image. 0-D int32 Tensor, Scalar

    Returns:
        bboxes (Tensor): The normalized proposal bounding boxes. 2-D float32 Tensor of
         shape (num_proposals, 4)
    """

    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

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


@tf.function
def roi_pooling(
    feature_map: Tensor,
    proposals: Tensor,
    width: Tensor,
    height: Tensor,
    pool_size: int = 7,
) -> Tensor:
    """Performs ROI pooling on the generated feature map using the proposed bounding
    boxes. This basically identifies regions of interest on the feature map using the
    cordinates in the proposals, and then, obtains features associated with these
    regions. This is done here by using a crop and resize operation from tensorflow.

    Args:
        feature_map (Tensor): The feature map generated with the backbone network. 4-D
         float32 Tensor usually of shape (1, height, widht, 1024)
        proposals (Tensor): The proposed bounding boxes from RPN. 2-D float32 Tensor of
         shape (num_proposals, 4)
        width (Tensor): The width of the image. 0-D int32 Tensor, Scalar
        height (Tensor): The height of the image. 0-D int32 Tensor, Scalar
        pool_size (int, optional): The final size (widht, height) to which to resize the
         obtained ROI. Defaults to 7.

    Returns:
        rois (Tensor): The regions of interest from the feature map, cropped and resized.
         4-D float32 Tensor of shape (num_proposals, pool_size, pool_size, 1024)
    """

    crop_size = pool_size * 2

    croped = tf.image.crop_and_resize(
        feature_map,
        normalize_bboxes(proposals, width, height),
        tf.cast(tf.zeros_like(proposals)[:, 0], tf.int32),
        [crop_size] * 2,
    )

    rois = tf.keras.layers.MaxPool2D()(croped)

    return rois
