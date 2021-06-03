import tensorflow as tf

from .bbox_utils import change_order
from .rpn_utils.proposals_utils import filter_proposals


def apply_nms(proposals, scores, nms_threshold: float, top_n: int):
    """Applies non-maximum suppression to proposals.

    Arguments:
        proposals: Tensor of shape (num_proposals, 4), holding the
            coordinates of the proposals' bounding boxes.
        scores: Tensor of shape (num_proposals,), holding the
            scores associated to each bounding box.

    Returns:
        (`proposals`, `scores`), but with NMS applied, and ordered by score.
    """
    selected_indces = tf.image.non_max_suppression(
        change_order(proposals), scores, top_n, nms_threshold
    )
    proposals = tf.gather(proposals, selected_indces, axis=0)
    scores = tf.gather(scores, selected_indces, axis=0)

    return proposals, scores


# per class non max suppression
def per_class_nms(
    bb,
    cls_,
    scores,
    im_size,
    score_thresh: float = 0.7,
    nms_threshold: float = 0.7,
    top_n: int = 19,
):
    uniques_clses, _ = tf.unique(cls_)
    s_bb = []
    s_scores = []

    for cl in uniques_clses:
        bb_ = tf.boolean_mask(bb, cls_ == cl)
        scor_ = tf.boolean_mask(scores, cls_ == cl)
        bb_, scor_ = filter_proposals(bb_, scor_, im_size, score_thresh, True)
        bb_, scor_ = apply_nms(bb_, scor_, nms_threshold, top_n)
        bb_ = tf.concat([bb_, tf.fill([len(bb_), 1], cl)], axis=-1)
        s_bb += [bb_]
        s_scores += [scor_]
    s_bb = tf.concat(s_bb, axis=0)
    s_scores = tf.concat(s_scores, axis=0)

    return s_bb, s_scores
