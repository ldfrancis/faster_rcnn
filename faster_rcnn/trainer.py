import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from .backbone.factory import get_backbone
from .frcnn import FRCNN
from .losses import smooth_l1_loss
from .utils.bbox_utils import decode, encode
from .utils.data_utils import create_data
from .utils.detector_targets_utils import generate_detector_targets
from .utils.nms_utils import apply_nms
from .utils.proposals_utils import filter_proposals
from .utils.roi_pooling_utils import roi_pooling
from .utils.rpn_anchor_utils import generate_anchors
from .utils.rpn_targets_utils import generate_rpn_targets


class Trainer:
    def __init__(self, frcnn: FRCNN):
        self.rpn = frcnn.rpn
        self.detector = frcnn.detector
        self.main_backbone = frcnn.backbone
        _, self.detector_backbone_head, _ = get_backbone(frcnn.cfg["backbone"])
        self.cfg = frcnn.cfg
        self.detector_optimizer = Adam(self.cfg["detector_lr"])
        self.backbone_head_optimizer = Adam(self.cfg["backbone_head_lr"])
        self.backbone_tail_optimizer = Adam(self.cfg["backbone_tail_lr"])
        self.detector_backbone_optimizer = Adam[self.cfg["backbone_head_lr"]]
        self.rpn_optimizer = Adam(self.cfg["rpn_lr"])

    def rpn_loss(self, anchor_targets, anchor_labels, rpn_deltas, rpn_score):
        filter_ = anchor_labels != -1  # not ignored
        valid_labels = tf.boolean_mask(anchor_labels, filter_)
        valid_rpn_score = tf.boolean_mask(rpn_score, filter_)

        # classification loss
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            valid_labels, valid_rpn_score
        )
        n_cls = valid_rpn_score.shape[0]

        # calculate regression loss only for proposals with positive labels
        filter_ = anchor_labels == 1  # positive labels
        valid_anchor_targets = tf.boolean_mask(anchor_targets, filter_)
        valid_rpn_deltas = tf.boolean_mask(rpn_deltas, filter_)

        # regression loss
        regr_loss = smooth_l1_loss(valid_rpn_deltas, valid_anchor_targets)
        n_reg = valid_anchor_targets.shape[0]

        total_loss = (1 / n_cls) * tf.reduce_sum(cls_loss) + 10 * (
            1 / n_reg
        ) * tf.reduce_sum(regr_loss)

        return total_loss

    def detector_loss(
        self, proposal_targets, proposal_labels, bbox_deltas, class_score
    ):
        # class loss
        # ignore targets less than 0 (-1) only consider fg and bg
        valid_filter = proposal_labels >= 0
        valid_proposal_labels = tf.boolean_mask(proposal_labels, valid_filter)
        valid_class_score = tf.boolean_mask(class_score, valid_filter)
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            valid_proposal_labels, valid_class_score
        )

        # bbox reg loss
        # ignore background boxes
        box_filter = proposal_labels > 0
        valid_proposal_targets = tf.boolean_mask(proposal_targets, box_filter)
        valid_bbox_deltas = tf.boolean_mask(bbox_deltas, box_filter)
        valid_proposal_labels = tf.boolean_mask(proposal_labels, box_filter)

        # labels to onehot, flatten bbox pred
        proposal_labels_onehot = tf.one_hot(
            valid_proposal_labels - 1, self.cfg["detector"]["num_classes"]
        )
        valid_bbox_deltas = tf.reshape(valid_bbox_deltas, [-1, 4])

        # obtain filter to select bbox preds
        bbox_pred_filter = tf.reshape(proposal_labels_onehot, [-1])
        bbox_pred_filter = tf.cast(bbox_pred_filter, tf.bool)
        valid_bbox_deltas = tf.boolean_mask(valid_bbox_deltas, bbox_pred_filter)

        # loss
        regr_loss = smooth_l1_loss(valid_bbox_deltas, valid_proposal_targets)

        return tf.reduce_sum(cls_loss) + tf.reduce_sum(regr_loss)

    def train(self, ds, v_ds=None):
        # 4-step alternating training
        self.ds = ds
        self.v_ds = v_ds
        if self.cfg["train_type"] == "4step":
            self.train4step()

    def train4step(self):
        rpn_stride = self.cfg["stride"]
        grad_clip = self.cfg["grad_clip"]
        bg_low = self.cfg["bg_low"]
        bg_high = self.cfg["bg_high"]
        fg_low = self.cfg["fg_low"]
        pos_prop_perc = self.cfg["pos_prop_perc"]
        prop_batch = self.cfg["prop_batch"]
        pool_size = self.cfg["pool_size"]
        margin = self.cfg["margin"]
        clobber_positive = self.cfg["clobber_positive"]
        neg_iou_thresh = self.cfg["neg_iou_thresh"]
        pos_iou_thresh = self.cfg["pos_iou_thresh"]
        pos_anchors_perc = self.cfg["pos_anchors_perc"]
        anchor_batch = self.cfg["anchor_batch"]
        epochs = self.cfg["epochs"]

        print("\nStep 1: Training rpn")
        for epoch in range(1, epochs + 1):
            rpnloss = tf.keras.metrics.Mean()
            for b, example in enumerate(self.ds):
                image, gt_bboxes = create_data(example)
                anchors = generate_anchors(
                    [image.shape[0] // rpn_stride, image.shape[1] // rpn_stride]
                )
                loss = self.train_rpn_step(
                    image,
                    gt_bboxes,
                    anchors,
                    grad_clip,
                    margin,
                    clobber_positive,
                    neg_iou_thresh,
                    pos_iou_thresh,
                    pos_anchors_perc,
                    anchor_batch,
                )
                rpnloss.update_state(loss)
                print(
                    f"\rEpoch {epoch}:\t batch {b} rpn loss: {rpnloss.result():.3f}",
                    end="",
                )

        print("\nStep 2: Training detector")
        for epoch in range(1, epochs + 1):
            detectorloss = tf.keras.metrics.Mean()
            for b, example in enumerate(self.ds):
                image, gt_bboxes = create_data(example)
                anchors = generate_anchors(
                    [image.shape[0] // rpn_stride, image.shape[1] // rpn_stride]
                )
                loss = self.train_detector_step(
                    image,
                    gt_bboxes,
                    anchors,
                    grad_clip,
                    bg_low,
                    bg_high,
                    fg_low,
                    pos_prop_perc,
                    prop_batch,
                    pool_size,
                )
                detectorloss.update_state(loss)
                print(
                    f"\rEpoch {epoch}:\t batch {b} detector loss: "
                    f"{detectorloss.result():.3f}",
                    end="",
                )

        print("\nStep 3: Training rpn with fixed base")
        self.main_backbone.head.set_weights(self.detector_backbone_head.get_weights())
        for epoch in range(1, epochs + 1):
            rpnloss = tf.keras.metrics.Mean()
            for b, example in enumerate(self.ds):
                image, gt_bboxes = create_data(example)
                anchors = generate_anchors(
                    [image.shape[0] // rpn_stride, image.shape[1] // rpn_stride]
                )
                loss = self.train_rpn_fb_step(
                    image,
                    gt_bboxes,
                    anchors,
                    grad_clip,
                    margin,
                    clobber_positive,
                    neg_iou_thresh,
                    pos_iou_thresh,
                    pos_anchors_perc,
                    anchor_batch,
                )
                rpnloss.update_state(loss)
                print(
                    f"\rEpoch {epoch}:\t batch {b} rpn loss: {rpnloss.result():.3f}",
                    end="",
                )

        print("\nStep 4: Training detector with fixed base; same base as rpn")
        for epoch in range(1, epochs + 1):
            detectorloss = tf.keras.metrics.Mean()
            for b, example in enumerate(self.ds):
                image, gt_bboxes = create_data(example)
                anchors = generate_anchors(
                    [image.shape[0] // rpn_stride, image.shape[1] // rpn_stride]
                )
                loss = self.train_detector_fb_step(
                    image,
                    gt_bboxes,
                    anchors,
                    grad_clip,
                    bg_low,
                    bg_high,
                    fg_low,
                    pos_prop_perc,
                    prop_batch,
                    pool_size,
                )
                detectorloss.update_state(loss)
                print(
                    f"\rEpoch {epoch}:\t batch {b} detector loss: "
                    f"{detectorloss.result():.3f}",
                    end="",
                )

    def train_rpn_fb_step(
        self,
        image,
        gt_bboxes,
        anchors,
        grad_clip,
        margin,
        clobber_positive,
        neg_iou_thresh,
        pos_iou_thresh,
        pos_anchors_perc,
        anchor_batch,
    ):
        x = tf.expand_dims(image, 0)
        with tf.GradientTape() as tape:
            im_shape = x.shape[1:3]  # H, W
            feat_map = self.main_backbone.head(x)
            rpn_deltas, rpn_scores = self.rpn(feat_map)
            rpn_targets, rpn_labels, _ = generate_rpn_targets(
                anchors,
                gt_bboxes,
                im_shape,
                margin,
                clobber_positive,
                neg_iou_thresh,
                pos_iou_thresh,
                pos_anchors_perc,
                anchor_batch,
            )
            # rpn loss
            rpnloss = self.rpn_loss(rpn_targets, rpn_labels, rpn_deltas, rpn_scores)
            total_loss = rpnloss

        rpn_grads = tape.gradient(total_loss, self.rpn.trainable_variables)

        # clip gradients
        rpn_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in rpn_grads]

        # apply grad
        self.rpn_optimizer.apply_gradients(zip(rpn_grads, self.rpn.trainable_variables))

        return total_loss.numpy()

    def train_rpn_step(
        self,
        image,
        gt_bboxes,
        anchors,
        grad_clip,
        margin,
        clobber_positive,
        neg_iou_thresh,
        pos_iou_thresh,
        pos_anchors_perc,
        anchor_batch,
    ):
        x = tf.expand_dims(image, 0)
        im_shape = x.shape[1:3]  # H, W
        with tf.GradientTape() as tape:
            feat_map = self.main_backbone.head(x)
            rpn_deltas, rpn_scores = self.rpn(feat_map)
            rpn_targets, rpn_labels, _ = generate_rpn_targets(
                anchors,
                gt_bboxes,
                im_shape,
                margin,
                clobber_positive,
                neg_iou_thresh,
                pos_iou_thresh,
                pos_anchors_perc,
                anchor_batch,
            )
            # rpn loss
            rpnloss = self.rpn_loss(rpn_targets, rpn_labels, rpn_deltas, rpn_scores)
            total_loss = rpnloss

        rpn_grads, base_grads = tape.gradient(
            total_loss,
            [self.rpn.trainable_variables, self.main_backbone.head.trainable_variables],
        )
        # clip gradients
        rpn_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in rpn_grads]
        base_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in base_grads]

        # apply grad
        self.rpn_optimizer.apply_gradients(zip(rpn_grads, self.rpn.trainable_variables))
        self.backbone_head_optimizer.apply_gradients(
            zip(base_grads, self.main_backbone.head.trainable_variables)
        )

        return total_loss.numpy()

    def train_detector_step(
        self,
        image,
        gt_bboxes,
        anchors,
        grad_clip,
        bg_low,
        bg_high,
        fg_low,
        pos_prop_perc,
        prop_batch,
        pool_size,
    ):
        x = tf.expand_dims(image, 0)
        im_shape = x.shape[1:3]  # H, W

        with tf.GradientTape() as tape:
            feat_map_det = self.detector_backbone_head(x)
            feat_map_rpn = self.main_backbone.head(x)

            rpn_deltas, rpn_scores = self.rpn(feat_map_rpn)

            # decode proposals
            rpn_proposals = decode(anchors, rpn_deltas)

            # filter and suppress proposals
            rpn_scores = rpn_scores[:, 1]
            rpn_proposals, rpn_scores = filter_proposals(
                rpn_proposals, rpn_scores, im_shape
            )
            rpn_proposals, rpn_scores = apply_nms(rpn_proposals, rpn_scores)

            # generate detector targets
            bbox_targets, bbox_labels, rpn_proposals = generate_detector_targets(
                rpn_proposals,
                gt_bboxes,
                im_shape,
                bg_low,
                bg_high,
                fg_low,
                pos_prop_perc,
                prop_batch,
            )

            # roi pooling
            rois = roi_pooling(
                feat_map_det, rpn_proposals, im_shape, pool_size=pool_size
            )

            # process rois
            rois = self.main_backbone.tail(rois)

            # detector prediction
            bbox_deltas, cls_score = self.detector(rois)

            # calculate detector loss
            detectorloss = self.detector_loss(
                bbox_targets, bbox_labels, bbox_deltas, cls_score
            )

        detector_grads, base_grads, tail_grads = tape.gradient(
            detectorloss,
            [
                self.detector.trainable_variables,
                self.detector_backbone_head.trainable_variables,
                self.main_backbone.tail.trainable_variables,
            ],
        )

        # clip gradients
        detector_grads = [
            tf.clip_by_value(g, -grad_clip, grad_clip) for g in detector_grads
        ]
        base_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in base_grads]
        tail_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in tail_grads]

        # apply grad
        self.detector_optimizer.apply_gradients(
            zip(detector_grads, self.detector.trainable_variables)
        )
        self.detector_backbone_optimizer.apply_gradients(
            zip(base_grads, self.detector_backbone_head.trainable_variables)
        )
        self.backbone_tail_optimizer.apply_gradients(
            zip(base_grads, self.main_backbone.tail.trainable_variables)
        )

        return detectorloss.numpy()

    def train_detector_fb_step(
        self,
        image,
        gt_bboxes,
        anchors,
        grad_clip,
        bg_low,
        bg_high,
        fg_low,
        pos_prop_perc,
        prop_batch,
        pool_size,
    ):
        x = tf.expand_dims(image, 0)
        im_shape = x.shape[1:3]  # H, W

        with tf.GradientTape() as tape:
            feat_map = self.main_backbone.head(x)

            rpn_deltas, rpn_scores = self.rpn(feat_map)

            # decode proposals
            rpn_proposals = decode(anchors, rpn_deltas)

            # filter and suppress proposals
            rpn_scores = rpn_scores[:, 1]
            rpn_proposals, rpn_scores = filter_proposals(
                rpn_proposals, rpn_scores, im_shape
            )
            rpn_proposals, rpn_scores = apply_nms(rpn_proposals, rpn_scores)

            # generate detector targets
            bbox_targets, bbox_labels, rpn_proposals = generate_detector_targets(
                rpn_proposals,
                gt_bboxes,
                im_shape,
                bg_low,
                bg_high,
                fg_low,
                pos_prop_perc,
                prop_batch,
            )

            # roi pooling
            rois = roi_pooling(feat_map, rpn_proposals, im_shape, pool_size=pool_size)

            # process rois
            rois = self.main_backbone.tail(rois)

            # detector prediction
            bbox_deltas, cls_score = self.detector(rois)

            # calculate detector loss
            detectorloss = self.detector_loss(
                bbox_targets, bbox_labels, bbox_deltas, cls_score
            )

        detector_grads, tail_grads = tape.gradient(
            detectorloss,
            [
                self.detector.trainable_variables,
                self.main_backbone.tail.trainable_variables,
            ],
        )

        # clip gradients
        detector_grads = [
            tf.clip_by_value(g, -grad_clip, grad_clip) for g in detector_grads
        ]
        tail_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in tail_grads]

        # apply grad
        self.detector_optimizer.apply_gradients(
            zip(detector_grads, self.detector.trainable_variables)
        )
        self.backbone_tail_optimizer.apply_gradients(
            zip(tail_grads, self.main_backbone.tail.trainable_variables)
        )

        return detectorloss.numpy()
