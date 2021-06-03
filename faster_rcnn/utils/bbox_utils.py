import tensorflow as tf


def get_dimensions_and_center(bboxes):
    """Obtain width, height and center coordinates of a bounding box.

    Arugments:
        bboxes: Tensor of shape (num_bboxes, 4).

    Returns:
        Tuple of Tensors of shape (num_bboxes,) with the values
        width, height, center_x and center_y corresponding to each
        bounding box.
    """
    x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)
    width = x2 - x1
    height = y2 - y1
    ctx = x1 + width / 2
    cty = y1 + height / 2

    return width, height, ctx, cty


def encode(anchors, bboxes):
    """Encode bounding boxes as deltas w.r.t. anchors.

    Arguments:
        anchors: Tensor of shape (num_bboxes, 4). With the same bbox
            encoding.
        bboxes: Tensor of shape (num_bboxes, 4). Having the bbox
            encoding in the (x_min, y_min, x_max, y_max) order.

    Returns:
        Tensor of shape (num_bboxes, 4) with the different deltas needed
            to transform `anchors` to `bboxes`. These deltas are with
            regard to the center, width and height of the two boxes.
    """
    anchors_w, anchors_h, anchors_ctx, anchors_cty = get_dimensions_and_center(anchors)
    bboxes_w, bboxes_h, bboxes_ctx, bboxes_cty = get_dimensions_and_center(bboxes)
    d_x = (bboxes_ctx - anchors_ctx) / anchors_w
    d_y = (bboxes_cty - anchors_cty) / anchors_h
    d_w = tf.math.log(bboxes_w / anchors_w)
    d_h = tf.math.log(bboxes_h / anchors_h)

    deltas = tf.concat([d_x, d_y, d_w, d_h], axis=-1)

    return deltas


def decode(anchors, deltas):
    """Decode bounding boxes by applying deltas to anchors.

    Arguments:
        anchors: Tensor of shape (num_bboxes, 4). Having the bbox
            encoding in the (x_min, y_min, x_max, y_max) order.
        deltas: Tensor of shape (num_bboxes, 4). Deltas (as returned by
            `encode`) that we want to apply to `bboxes`.

    Returns:
        Tensor of shape (num_bboxes, 4) with the decoded proposals,
            obtained by applying `deltas` to `anchors`.
    """
    anchors_w, anchors_h, anchors_ctx, anchors_cty = get_dimensions_and_center(anchors)
    d_x, d_y, d_w, d_h = tf.split(deltas, 4, axis=-1)
    bboxes_w = tf.exp(d_w) * anchors_w
    bboxes_h = tf.exp(d_h) * anchors_h
    bboxes_ctx = anchors_w * d_x + anchors_ctx
    bboxes_cty = anchors_h * d_y + anchors_cty
    x1 = bboxes_ctx - bboxes_w / 2
    x2 = bboxes_ctx + bboxes_w / 2
    y1 = bboxes_cty - bboxes_h / 2
    y2 = bboxes_cty + bboxes_h / 2
    bboxes = tf.concat([x1, y1, x2, y2], axis=-1)

    return bboxes


def bbox_overlap_tf(bboxes1, bboxes2):
    """Calculate Intersection over Union (IoU) between two sets of bounding
    boxes.
    Args:
        bboxes1: shape (total_bboxes1, 4)
            with x1, y1, x2, y2 point order.
        bboxes2: shape (total_bboxes2, 4)
            with x1, y1, x2, y2 point order.
        p1 *-----
           |     |
           |_____* p2
    Returns:
        Tensor with shape (total_bboxes1, total_bboxes2)
        with the IoU (intersection over union) of bboxes1[i] and bboxes2[j]
        in [i, j].
    """
    with tf.name_scope("bbox_overlap"):
        x11, y11, x12, y12 = tf.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = tf.split(bboxes2, 4, axis=1)

        xI1 = tf.maximum(x11, tf.transpose(x21))
        yI1 = tf.maximum(y11, tf.transpose(y21))

        xI2 = tf.minimum(x12, tf.transpose(x22))
        yI2 = tf.minimum(y12, tf.transpose(y22))

        intersection = tf.maximum(xI2 - xI1 + 1.0, 0.0) * tf.maximum(
            yI2 - yI1 + 1.0, 0.0
        )

        bboxes1_area = (x12 - x11 + 1) * (y12 - y11 + 1)
        bboxes2_area = (x22 - x21 + 1) * (y22 - y21 + 1)

        union = (bboxes1_area + tf.transpose(bboxes2_area)) - intersection

        iou = tf.maximum(intersection / union, 0)

        return iou


def change_order(bboxes):
    """Change bounding box encoding order.

    Tensorflow works with the (y_min, x_min, y_max, x_max) order while we work
    with the (x_min, y_min, x_max, y_min).

    While both encoding options have its advantages and disadvantages we
    decided to use the (x_min, y_min, x_max, y_min), forcing us to switch to
    Tensorflow's every time we want to use function that handles bounding
    boxes.

    Arguments:
        bboxes: A Tensor of shape (total_bboxes, 4).

    Returns:
        bboxes: A Tensor of shape (total_bboxes, 4) with the order swaped.
    """
    v1, v2, v3, v4 = tf.unstack(bboxes, axis=-1)

    bboxes = tf.stack([v2, v1, v4, v3], axis=1)

    return bboxes
