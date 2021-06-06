from typing import Any, Dict

import tensorflow as tf
from tensorflow import Tensor

from .backbone.factory import get_backbone
from .detection import Detector
from .rpn import RPN
from .utils.bbox_utils import decode, encode
from .utils.data_utils.tfds_utils import display_image
from .utils.nms_utils import apply_nms, per_class_nms
from .utils.rpn_utils.proposals_utils import filter_proposals
from .utils.rpn_utils.roi_pooling_utils import roi_pooling
from .utils.rpn_utils.rpn_anchor_utils import generate_anchors


class FRCNN:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.backbone, _, _ = get_backbone(config["backbone"])
        self.rpn = RPN(config["rpn"])
        self.detector = Detector(config["detector"])

    def __call__(self, x: Tensor):
        if len(x.shape) == 3:
            x = tf.expand_dims(x, 0)
        elif x.shape[0] > 1:
            raise Exception("only a single image can be operated on")
        elif len(x.shape) > 4 or len(x.shape) < 3:
            raise Exception("only a 3D or 4D tensor is allowed")

        im_shape = x.shape[1:3]  # H, W
        feat_map = self.backbone(x)
        rpn_deltas, rpn_scores = self.rpn(feat_map)
        anchors = generate_anchors(feat_map.shape)

        # decode proposals
        rpn_proposals = decode(anchors, rpn_deltas)

        # filter and suppress proposals
        rpn_scores = rpn_scores[:, 1]
        rpn_proposals, rpn_scores = filter_proposals(
            rpn_proposals, rpn_scores, im_shape, self.rpn.cfg["score_thresh"], True
        )
        rpn_proposals, rpn_scores = apply_nms(
            rpn_proposals,
            rpn_scores,
            self.rpn.cfg["nms_threshold"],
            self.rpn.cfg["top_n"],
        )

        # roi pooling
        rois = roi_pooling(
            feat_map, rpn_proposals, im_shape, pool_size=self.rpn.cfg["pool_size"]
        )

        # process roi features
        rois_feats = self.backbone(rois, part="tail")

        # detector prediction
        bbox_deltas, cls_score = self.detector(rois_feats)

        # obtain class labels and bounding boxes
        cls_label = tf.argmax(cls_score, axis=1) - 1
        cls_label_onehot = tf.one_hot(cls_label, self..detector.cfg["num_classes"])
        deltas = tf.boolean_mask(
            tf.reshape(bbox_deltas, [-1, 4]), tf.reshape(cls_label_onehot, [-1])
        )
        scores_ = tf.boolean_mask(
            tf.reshape(cls_score[:, 1:], [-1]), tf.reshape(cls_label_onehot, [-1])
        )
        lbl = tf.boolean_mask(cls_label, cls_label > 0)
        lbl = tf.cast(lbl, tf.float32)

        boxes_ = tf.boolean_mask(rpn_proposals, cls_label >= 0)
        bboxes_ = decode(boxes_, deltas)

        # filter class bounding boxes with non max suppression
        s_bb, s_scores = None, None
        if len(bboxes_) > 0:
            s_bb, s_scores = per_class_nms(
                bboxes_,
                lbl,
                scores_,
                im_shape,
                self.cfg["detector"]["score_thresh"],
                self.cfg["detector"]["nms_threshold"],
                self.cfg["detector"]["top_n"],
            )

        return s_bb, s_scores
