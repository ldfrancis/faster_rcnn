import numpy as np
import tensorflow as tf


def sort_anchors(anchors):
    """Sort the anchor references aspect ratio first, then area."""
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]

    aspect_ratios = np.round(heights / widths, 1)
    areas = widths * heights

    return anchors[np.lexsort((areas, aspect_ratios)), :]


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """Generate base set of anchors to be used as reference for all anchors.
    Anchors vary only in width and height. Using the base_size and the
    different ratios we can calculate the desired widths and heights.
    Aspect ratios maintain the area of the anchors, while scales apply to the
    length of it (and thus affect it squared).
    Arguments:
        base_size (int): Base size of the base anchor (square).
        aspect_ratios: Ratios to use to generate different anchors. The ratio
            is the value of height / width.
        scales: Scaling ratios applied to length.
    Returns:
        anchors: Numpy array with shape (total_aspect_ratios * total_scales, 4)
            with the corner points of the reference base anchors using the
            convention (x_min, y_min, x_max, y_max).
    """
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack(
        [
            center_xy - widths / 2,
            center_xy - heights / 2,
            center_xy + widths / 2,
            center_xy + heights / 2,
        ]
    )

    return sort_anchors(anchors)


def generate_anchors(feature_map_shape, config):
    """Generate anchors for an image.

    Using the feature map (the output of the pretrained network for an image)
    and the anchor references (generated using the specified anchor sizes and
    ratios), we generate a list of anchors.

    Anchors are just fixed bounding boxes of different ratios and sizes that
    are uniformly generated throughout the image.

    Arguments:
        feature_map_shape: Shape of the convolutional feature map used as
            input for the RPN.
            Should be (batch, feature_height, feature_width, depth).

    Returns:
        all_anchors: A Tensor with the anchors at every spatial position, of
            shape `(feature_height, feature_width, num_anchors_per_points, 4)`
            using the (x1, y1, x2, y2) convention.
    """

    anchor_reference = generate_anchors_reference(
        config["anchor_base_size"], config["anchor_ratios"], config["anchor_scales"]
    )

    h, w = feature_map_shape[1:3]
    h_array = np.arange(h, dtype=np.int32)  # array starting from 0 to h
    w_array = np.arange(w, dtype=np.int32)  # array starting form 0 to w
    w_wise, h_wise = np.meshgrid(w_array, h_array)
    w_wise = w_wise.reshape((-1, 1)) * 16
    h_wise = h_wise.reshape((-1, 1)) * 16
    x1 = w_wise + anchor_reference[:, 0]
    x2 = w_wise + anchor_reference[:, 2]
    y1 = h_wise + anchor_reference[:, 1]
    y2 = h_wise + anchor_reference[:, 3]
    all_anchors = np.stack([x1, y1, x2, y2], axis=-1)
    _, _, p = all_anchors.shape
    all_anchors = all_anchors.reshape((-1, p))
    all_anchors = tf.constant(all_anchors, dtype=tf.float32)

    return all_anchors
