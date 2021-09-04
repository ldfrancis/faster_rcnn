from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor


@tf.function(experimental_relax_shapes=True)
def to_center_width_height(bboxes: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Computes the center coordinates, width, and height of a bounding box from its
    corners coordinates x1,y1,x2,y2

    Args:
        bboxes (Tensor): The bounding boxes for which to compute the center, width and
        height. 2-D float32 Tensor of shape (num_boxes, 4)

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: Tensors containing the center_x,
        center_y, width and height. Each of shape (num_boxes,)
    """
    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    return center_x, center_y, width, height


@tf.function(experimental_relax_shapes=True)
def encode(bboxes: Tensor, gt_bboxes: Tensor) -> Tensor:
    """Encodes the ground-truth bounding boxes wrt to another Tensor of bounding boxes
    (This could be anchors or rpn proposed boxes) to geenerate the targets for bounding
    box regression. This targets are the results of a scale-invariant translation of the
    center of a bounding box and a log-space translation of its width and height.

    Args:
        bboxes (Tensor): The bounding boxes to generate regression targets for. 2-D
        float32 Tensor.
        gt_bboxes (Tensor): The ground-truth bounding boxes. 2-D float32 Tensor

    Returns:
        Tensor: The encoded targets for rpn bounding box regression. 2-D float32 Tensor
    """
    bboxes_center_x, bboxes_center_y, bboxes_w, bboxes_h = to_center_width_height(
        bboxes
    )
    (
        gt_bboxes_center_x,
        gt_bboxes_center_y,
        gt_bboxes_w,
        gt_bboxes_h,
    ) = to_center_width_height(gt_bboxes)

    d_x = (bboxes_center_x - gt_bboxes_center_x) / bboxes_w
    d_y = (bboxes_center_y - gt_bboxes_center_y) / bboxes_h
    d_w = tf.math.log(gt_bboxes_w / bboxes_w)
    d_h = tf.math.log(gt_bboxes_h / bboxes_h)

    return tf.concat([d_x, d_y, d_w, d_h], axis=1)


@tf.function(experimental_relax_shapes=True)
def decode(bboxes: Tensor, deltas: Tensor) -> Tensor:
    """Decodes the predicted bounding box regression deltas to obtain the expected
    ground-truth bounding boxes.

    Args:
        bboxes (Tensor): Usually the anchor boxes or rpn proposed boxes. 2-D float32
        Tensor
        deltas (Tensor): The output of bounding box regression. 2-D float32 Tensor

    Returns:
        Tensor: The expected ground-truth boxes decoded from deltas
        (output of bounding box regression). 2-D float32 Tensor
    """
    bboxes_center_x, bboxes_center_y, bboxes_w, bboxes_h = to_center_width_height(
        bboxes
    )
    d_x, d_y, d_w, d_h = tf.split(deltas, 4, axis=1)

    gt_bboxes_w = tf.exp(d_w) * bboxes_w
    gt_bboxes_h = tf.exp(d_h) * bboxes_h
    gt_bboxes_center_x = bboxes_w * d_x + bboxes_center_x
    gt_bboxes_center_y = bboxes_h * d_y + bboxes_center_y

    x1 = gt_bboxes_center_x - gt_bboxes_w / 2
    y1 = gt_bboxes_center_y - gt_bboxes_h / 2
    x2 = gt_bboxes_center_x + gt_bboxes_w / 2
    y2 = gt_bboxes_center_y + gt_bboxes_h / 2

    return tf.concat([x1, y1, x2, y2], axis=1)


@tf.function
def bbox_overlap(bboxes1: Tensor, bboxes2: Tensor) -> Tensor:
    """Calculates the overlap/iou(intersection over union) between 2 sets of bounding
    boxes. For each box in bboxes1, its overlap with all the boxes in bboxes2 is
    calculated

    Args:
        bboxes1 (Tensor): The first set of bounding boxes. 2-D float32 Tensor of shape
        (num_bboxes1, 4)
        bboxes2 (Tensor): The second set of bounding boxes. 2-D float32 Tensor of shape
        (num_bboxes2, 4)

    Returns:
        Tensor: A Tensor containing the overlap/iou of each box in bboxes1 with all boxes
        in bboxes2. 2-D float32 Tensor os shape (num_bboxes1, num_bboxes2)
    """

    bboxes1_x1, bboxes1_y1, bboxes1_x2, bboxes1_y2 = tf.split(bboxes1, 4, axis=1)
    bboxes2_x1, bboxes2_y1, bboxes2_x2, bboxes2_y2 = tf.split(
        tf.transpose(bboxes2), 4, axis=0
    )

    innerx1 = tf.maximum(bboxes1_x1, bboxes2_x1)
    innery1 = tf.maximum(bboxes1_y1, bboxes2_y1)
    innerx2 = tf.minimum(bboxes1_x2, bboxes2_x2)
    innery2 = tf.minimum(bboxes1_y2, bboxes2_y2)

    inner_width = tf.maximum(innerx2 - innerx1 + 1, 0)
    inner_height = tf.maximum(innery2 - innery1 + 1, 0)

    inner_area = inner_width * inner_height
    bboxes1_area = (bboxes1_x2 - bboxes1_x1 + 1) * (bboxes1_y2 - bboxes1_y1 + 1)
    bboxes2_area = (bboxes2_x2 - bboxes2_x1 + 1) * (bboxes2_y2 - bboxes2_y1 + 1)

    intersection = inner_area
    union = bboxes1_area + bboxes2_area - inner_area

    iou = tf.maximum(intersection / union, 0)

    return iou


@tf.function
def swap_xy(bboxes: Tensor) -> Tensor:
    """Changes the order of a bounding box's cordinates x1,y1 and x2,y2 to y1,x1 and
    y2,x2. This is usually done to conform to the convention used in tensorflow.
    Tensorflow uses y1,x1,y2,x2

    Args:
        bboxes (Tensor): The bounding boxes. 2-D float32 Tensor of shape (num_bboxes,4)

    Returns:
        Tensor: The bounding boxes with swapped x and y. 2-D float32 Tensor
    """
    x1, y1, x2, y2 = tf.unstack(bboxes, axis=-1)
    bboxes = tf.stack([y1, x1, y2, x2], axis=1)

    return bboxes
