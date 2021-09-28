import numpy as np
import tensorflow as tf
from tensorflow import Tensor


  
def generate_reference_anchors(
    base_size: Tensor, scales: Tensor, aspect_ratios: Tensor
) -> Tensor:
    """Generate a tensor of anchors given the base size, scales, and aspect ratios.
    The would serve as the anchor boxes at position 0,0 (top-left corner) of an image.
    Anchor boxes at other positions would be generated using this as reference.

    Args:
        base_size (Tensor): The size of an anchor box dimension for unit scale and
        unit aspect ratio
        scales (Tensor): The scales applied to the anchor box dimesions
        aspect_ratios (Tensor): Aspect ratios for anchor boxes

    Returns:
        Tensor: The generated anchor boxes
    """
    scales_grid, aspect_grid = tf.meshgrid(scales, aspect_ratios)
    aspect_grid_sqrt = tf.math.sqrt(aspect_grid)
    base_size = tf.cast(base_size, dtype=scales_grid.dtype)

    w = scales_grid * base_size * aspect_grid_sqrt
    h = scales_grid * base_size / aspect_grid_sqrt

    x1 = -w / 2
    y1 = -h / 2
    x2 = w / 2
    y2 = h / 2

    reference_anchors = tf.stack([x1, y1, x2, y2], axis=-1)
    reference_anchors = tf.reshape(reference_anchors, [-1, 4])

    return reference_anchors


  
def generate_anchors(
    feat_map: Tensor,
    base_size: Tensor,
    stride: Tensor,
    scales: Tensor,
    aspect_ratios: Tensor,
) -> Tensor:
    """Generates all anchors for each position in a feature map using the reference
    anchors.

    Args:
        feat_map (Tensor): A feature map from a CNN. float32 4-D Tensor
        base_size (Tensor): The base size of the reference anchors for unit scale
        and aspect ratio. int32 0-D Tensor, Scalar
        stride (Tensor): This could size reduction factor from the input image to
        the feature map or the amount of shift to incorporate when selecting positions
        in the input image to generate anchors for. int32 0-D Tensor, Scalar
        scales (Tensor): The scale by which to increase the reference anchor boxes
        dimensios. float32 1-D Tensor
        aspect_ratios (Tensor): The aspect ratios for the reference anchor boxes.
        float32. 1-D Tensor

    Returns:
        Tensor: The generated anchor boxes. float32 2-D Tensor
    """
    _, H, W, _ = feat_map.shape
    row_positions, col_positions = tf.meshgrid(
        tf.range(W) * stride, tf.range(H) * stride
    )
    row_positions = tf.reshape(row_positions, [-1])
    col_positions = tf.reshape(col_positions, [-1])
    positions = tf.stack(
        [row_positions, col_positions, row_positions, col_positions], axis=1
    )
    positions = tf.expand_dims(positions, axis=1)
    positions = tf.cast(positions, tf.float32)

    reference_anchors = generate_reference_anchors(base_size, scales, aspect_ratios)
    reference_anchors = tf.expand_dims(reference_anchors, axis=0)

    anchors = positions + reference_anchors
    anchors = tf.reshape(anchors, [-1, 4])

    return anchors
