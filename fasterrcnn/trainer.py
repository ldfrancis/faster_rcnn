from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras.optimizers import Adam

from .backbone.factory import get_backbone
from .frcnn import FRCNN
from .losses import smooth_l1_loss
from .utils.bbox_utils import decode
from .utils.checkpoint_utils import create_checkpoint
from .utils.data_utils import create_data
from .utils.detector_targets_utils import generate_detector_targets
from .utils.logging_utils import TensorboardLogger
from .utils.nms_utils import apply_nms
from .utils.path_utils import resolve_path
from .utils.proposals_utils import filter_proposals
from .utils.roi_pooling_utils import roi_pooling
from .utils.rpn_anchors_utils import generate_anchors
from .utils.rpn_targets_utils import generate_rpn_targets


class Trainer:
    def __init__(self, frcnn: FRCNN, trainer_cfg: Dict[str, Any] = {}) -> None:
        """Handle the training of a faster rcnn model

        Args:
            frcnn (FRCNN): A fasterrcnn model
            trainer_cfg (Dict[str, Any], optional): Config containing settings to be
            used for training. Defaults to {}.
        """
        self.rpn = frcnn.rpn
        self.detector = frcnn.detector
        self.main_backbone = frcnn.backbone
        _, self.detector_backbone_head, _ = get_backbone(frcnn.cfg["rpn"]["backbone"])
        self.cfg = {**frcnn.cfg, **trainer_cfg}
        self.detector_optimizer = Adam(float(self.cfg["detector_lr"]))
        self.backbone_head_optimizer = Adam(float(self.cfg["backbone_head_lr"]))
        self.backbone_tail_optimizer = Adam(float(self.cfg["backbone_tail_lr"]))
        self.detector_backbone_optimizer = Adam(float(self.cfg["backbone_head_lr"]))
        self.rpn_optimizer = Adam(float(self.cfg["rpn_lr"]))

        self.best_score = tf.Variable(np.inf, dtype=tf.float32)
        self.eval_loss = np.inf

        self.ckpt = None
        self.checkpoint_path = None
        self.ckpt_manager = None

        self.logger = TensorboardLogger(self.cfg)
        self.trainer_step = tf.Variable(1)

    def create_checkpoint(self):
        """Checkpoint for saving train progress and enabling training resumption"""
        max_to_keep = self.cfg.get("max_to_keep", 5)
        self.checkpoint_path = (
            resolve_path("~/faster_rcnn/checkpoints") / self.cfg["experiment_name"]
        )
        self.checkpoint = create_checkpoint(
            checkpoint_path=self.checkpoint_path,
            max_to_keep=max_to_keep,
            rpn=self.rpn,
            detector=self.detector,
            backbone_head=self.main_backbone.head,
            backbone_tail=self.main_backbone.tail,
            detector_optimizer=self.detector_optimizer,
            rpn_optimizer=self.rpn_optimizer,
            detector_backbone_optimizer=self.detector_backbone_optimizer,
            backbone_head_optimizer=self.backbone_head_optimizer,
            backbone_tail_optimizer=self.backbone_tail_optimizer,
            best_score=self.best_score,
            trainer_step=self.trainer_step,
            metric=self.evaluation_metric,
            train_type=tf.Variable(self.cfg["train_type"]),
        )

        self.logger.warning(f"Created checkpoint at {self.checkpoint_path.absolute()}")

        restore_path = self.cfg.get("restore")
        if restore_path:
            self.checkpoint.restore(restore_path)
            if self.checkpoint.train_type != self.cfg["train_type"]:
                self.logger.warning(
                    "Could not restore checkpoint. train_type mismatch. loaded "
                    f"checkpoint is of train_type,{self.checkpoint.train_type}, "
                    "while the trainer was configured with train_type,"
                    f"{self.cfg['train_type']}"
                )
            else:
                self.logger.warning(f"Restored checkpoint from {restore_path}")

    def rpn_loss(
        self,
        anchor_targets: Tensor,
        anchor_labels: Tensor,
        rpn_deltas: Tensor,
        rpn_score: Tensor,
    ) -> Tensor:
        """Compute the loss of the RPN

        Args:
            anchor_targets (Tensor): The targets for anchor bounding box regression.
             2-D float32 Tensor of shape (num_anchors, 4)
            anchor_labels (Tensor): The labels for the achors in {0, 1, or -1}. 1-D
             int32 Tensor of shape (num_anchors,)
            rpn_deltas (Tensor): The predicted offsets/deltas for anchor bounding box
             regression. 2-D float32 Tensor of shape (num_anchors, 4)
            rpn_score (Tensor): The predicted scores in [0,1]. 1-D float32 Tensor of
             shape (num_anchors,)

        Returns:
            Tensor: The output loss. 0-D float32 Tensor, Scalar
        """
        filter_ = anchor_labels != -1  # not ignored
        valid_labels = tf.boolean_mask(anchor_labels, filter_)

        valid_rpn_score = tf.boolean_mask(rpn_score, filter_)

        # classification loss
        cls_loss = tf.keras.losses.sparse_categorical_crossentropy(
            valid_labels, valid_rpn_score
        )
        n_cls = tf.reduce_sum(tf.cast(filter_, tf.float32))

        # calculate regression loss only for proposals with positive labels
        filter_ = anchor_labels == 1  # positive labels
        valid_anchor_targets = tf.boolean_mask(anchor_targets, filter_)
        valid_rpn_deltas = tf.boolean_mask(rpn_deltas, filter_)

        # regression loss
        regr_loss = smooth_l1_loss(valid_rpn_deltas, valid_anchor_targets)
        n_reg = tf.reduce_sum(tf.cast(filter_, tf.float32))

        total_loss = (1 / n_cls) * tf.reduce_sum(cls_loss) + 10 * (
            1 / n_reg
        ) * tf.reduce_sum(regr_loss)

        return total_loss

    def detector_loss(
        self,
        proposal_targets: Tensor,
        proposal_labels: Tensor,
        bbox_deltas: Tensor,
        class_score: Tensor,
    ) -> Tensor:
        """Calculate the detector loss

        Args:
            proposal_targets (Tensor): The targets for proposal bounding box regression
             2-D float32 Tensor of shape (num_proposals,4)
            proposal_labels (Tensor): The labels in {-1,0,1,..,num_classes} for the
             proposed bounding boxes. -1-> invalid box, 0 -> background,
             1,...,num_classes->foreground objects. 1-D int32 Tensor of shape
             (num_proposals,)
            bbox_deltas (Tensor): The predicted offsets/deltas by the detector. 2-D
             float32 Tensor of shape (num_proposals, 4*num_classes)
            class_score (Tensor): The predicted class score by the detector. 2-D float32
             Tensor of shape (num_proposals, num_classes+1)

        Returns:
            Tensor: The output loss. 0-D float32 Tensor, Scalar
        """
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

    def train(self, dataset: Iterator, valid_dataset: Iterator = None):
        """Train the frcnn model using a dataset and based on the training procedure
        (4 step Alternating training or approximate joint training) described in the
        paper


        Args:
            dataset (Iterator): The train dataset. An iterator that yields a dictionary,
             example, that contains the keys, "image" and "objects". The value for
             "image" is the input image while "objects" is a dictionary containt the
             bounding boxes and the labels. The bounding boxes, a 2-D array of shape
             (num_boxes, 4) is the value of the "box" key while labels a 1-D array of
             shape (num_boxes) is the value of the "labels" key

            valid_dataset (Iterator): The validation dataset with the same format as the
            train dataset
        """
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.evaluation_metric = self.cfg.get("evaluation_metric", "loss")
        self.create_checkpoint()

        if self.cfg["train_type"] == "4step":
            self.step_method_map = {
                1: self.train_rpn_step,
                2: self.train_detector_step,
                3: self.train_rpn_fb_step,
                4: self.train_detector_fb_step,
            }
            self.forward_method_map = {
                1: self.rpn_forward_step,
                2: self.detector_forward_step,
                3: self.rpn_forward_step,
                4: self.detector_forward_fb_step,
            }
            self.train4step()
        elif self.cfg["train_type"] == "approximate":
            self.train_approximate()

    def compute_evaluation_loss(self, step_function):
        assert self.valid_dataset is not None
        mean_loss = tf.keras.metrics.Mean()
        for b, example in enumerate(self.valid_dataset):
            image, gt_bboxes = create_data(example, self.cfg["image_base_size"])
            x = tf.expand_dims(image, 0)
            H, W = x.shape[1:3]
            im_size = tf.constant([H, W])  # H, W
            loss = step_function(image, gt_bboxes)
            mean_loss.update_state(loss)
        if self.eval_loss > mean_loss.result().numpy():
            self.eval_loss = mean_loss.result().numpy()
            self.patience = 5
        else:
            self.patience -= 1
            
        return self.eval_loss, mean_loss.result().numpy(), self.patience

    def reset_eval_loss(self):
        self.eval_loss = np.inf
        self.patience = 5

    def train_approximate(self):
        """Train using the approximate joint training procedure"""
        epochs = self.cfg["epochs"]
        for epoch in range(1, epochs + 1):
            mean_loss = tf.keras.metrics.Mean()
            for b, example in enumerate(self.dataset):
                image, gt_bboxes = create_data(example, self.cfg["image_base_size"])
                loss = self.train_approximate_step(
                    image,
                    gt_bboxes,
                )
                mean_loss.update_state(loss)
                
                if self.logger.has_message_time_elapsed():
                    best_eval_loss, eval_loss, patience = self.compute_evaluation_loss(self.forward_method_map[step])
                    if self.patience <= 0:
                        end_epoch = True

                self.logger.log(
                    {
                        "trainer_step": 1,
                        "loss": loss,
                        "mean_loss": mean_loss.result(),
                        "best_eval_loss": best_eval_loss,
                        "eval_loss": eval_loss,
                        "patience": patience,
                        "epoch": epoch,
                        "batch": b,
                    }
                )
            score = (
                self.eval_loss
            )
            if self.checkpoint.update(score):
                self.logger.warning(
                    f"Saving new checkpoint at epoch {epoch} for train type "
                    f"{self.checkpoint.train_type} with {self.checkpoint.metric} "
                    f"of {self.checkpoint.best_score}"
                )

    def approximate_forward_step(self, image, im_size, gt_bboxes):
        """Forward from backbone to rpn to detector

        Args:
            image (Tensor): Input image, 4-D float32 Tensor of shape (1,H,W,3)
            im_size: size of image 1-D tensor of shape (2,)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        feat_map = self.main_backbone.head(image)
        rpn_deltas, rpn_scores = self.rpn(feat_map)
        anchors = generate_anchors(
            feat_map,
            tf.constant(self.cfg["rpn"]["anchor_base_size"]),
            tf.constant(self.cfg["rpn"]["stride"]),
            tf.constant(self.cfg["rpn"]["anchor_scales"]),
            tf.constant(self.cfg["rpn"]["anchor_ratios"]),
        )
        rpn_targets, rpn_labels = generate_rpn_targets(
            anchors,
            gt_bboxes,
            im_size,
            tf.constant(self.cfg["margin"]),
            tf.constant(self.cfg["clobber_positive"]),
            tf.constant(self.cfg["neg_iou_thresh"]),
            tf.constant(self.cfg["pos_iou_thresh"]),
            tf.constant(self.cfg["pos_anchors_perc"]),
            tf.constant(self.cfg["anchor_batch"]),
        )
        # rpn loss
        rpnloss = self.rpn_loss(rpn_targets, rpn_labels, rpn_deltas, rpn_scores)

        # decode proposals
        rpn_proposals = decode(anchors, rpn_deltas)

        # filter and suppress proposals
        rpn_scores = rpn_scores[:, 1]
        rpn_proposals, rpn_scores = filter_proposals(
            rpn_proposals, rpn_scores, im_size
        )

        rpn_proposals, rpn_scores = apply_nms(
            rpn_proposals,
            rpn_scores,
            tf.constant(self.cfg["rpn"]["nms_threshold"], tf.float32),
            tf.constant(self.cfg["rpn"]["top_n"], tf.int32),
        )

        # generate detector targets
        bbox_targets, bbox_labels = generate_detector_targets(
            rpn_proposals,
            gt_bboxes,
            tf.constant(self.cfg["bg_low"], tf.float32),
            tf.constant(self.cfg["bg_high"], tf.float32),
            tf.constant(self.cfg["fg_low"], tf.float32),
            tf.constant(self.cfg["pos_prop_perc"], tf.float32),
            tf.constant(self.cfg["prop_batch"], tf.int32),
        )

        # roi pooling
        rois = roi_pooling(
            feat_map,
            rpn_proposals,
            W,
            H,
            pool_size=tf.constant(self.cfg["pool_size"], tf.int32),
        )

        # process rois
        rois = self.main_backbone.tail(rois)

        # detector prediction
        bbox_deltas, cls_score = self.detector(rois)

        # calculate detector loss
        detectorloss = self.detector_loss(
            bbox_targets, bbox_labels, bbox_deltas, cls_score
        )

        total_loss = detectorloss + rpnloss
        
        return total_loss

    
    def train_approximate_step(
        self,
        image: Tensor,
        gt_bboxes: Tensor,
    ) -> Tensor:
        """Train using the approximate joint training for just one step

        Args:
            image (Tensor): Input image. 2-D float32 Tensor of shape (H,W,3)
            gt_bboxes (Tensor): The ground truth bounding boxes. 2-D float32 Tensor
             of shape (num_boxes, 5)

        Returns:
            Tensor: The output loss. 0-D float32 Tensor
        """
        x = tf.expand_dims(image, 0)
        H, W = x.shape[1:3]
        im_size = tf.constant([H, W])  # H, W

        with tf.GradientTape() as tape:
            total_loss = approximate_forward_step(x, im_size, gt_bboxes)

        (
            rpn_grads,
            backbone_head_grads,
            backbone_tail_grads,
            detector_grads,
        ) = tape.gradient(
            total_loss,
            [
                self.rpn.trainable_variables,
                self.main_backbone.head.trainable_variables,
                self.main_backbone.tail.trainable_variables,
                self.detector.trainable_variables,
            ],
        )

        # clip gradients
        grad_clip = self.cfg["grad_clip"]
        rpn_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in rpn_grads]
        backbone_head_grads = [
            tf.clip_by_value(g, -grad_clip, grad_clip) for g in backbone_head_grads
        ]
        backbone_tail_grads = [
            tf.clip_by_value(g, -grad_clip, grad_clip) for g in backbone_tail_grads
        ]
        detector_grads = [
            tf.clip_by_value(g, -grad_clip, grad_clip) for g in detector_grads
        ]

        # apply grad
        self.rpn_optimizer.apply_gradients(zip(rpn_grads, self.rpn.trainable_variables))
        self.backbone_head_optimizer.apply_gradients(
            zip(backbone_head_grads, self.main_backbone.head.trainable_variables)
        )
        self.backbone_tail_optimizer.apply_gradients(
            zip(backbone_tail_grads, self.main_backbone.tail.trainable_variables)
        )
        self.detector_optimizer.apply_gradients(
            zip(detector_grads, self.detector.trainable_variables)
        )

        return total_loss

    def train4step(self):
        """Train the faster rcnn model using the 4 step alternating training described
        in the paper
        """
        epochs = self.cfg["epochs"]
        base_size = self.cfg["image_base_size"]

        def _run_trainer_step(step: int, message: str):
            self.logger.warning(f"\nStep {step}: {message}")
            end_epoch = False
            for epoch in range(1, epochs + 1):
                mean_loss = tf.keras.metrics.Mean()
                for b, example in enumerate(self.dataset):
                    image, gt_bboxes = create_data(example, base_size)
                    loss = self.step_method_map[step](image, gt_bboxes)
                    mean_loss.update_state(loss)
                    
                    if self.logger.has_message_time_elapsed():
                        best_eval_loss, eval_loss, patience = self.compute_evaluation_loss(self.forward_method_map[step])
                        if self.patience <= 0:
                            end_epoch = True
                        
                        self.logger.log(
                            {
                                "trainer_step": step,
                                "loss": loss,
                                "mean_loss": mean_loss.result(),
                                "best_eval_loss": best_eval_loss,
                                "eval_loss": eval_loss,
                                "patience": patience,
                                "epoch": epoch,
                                "batch": b,
                            }
                        )
                    

                if step in [2, 4]:
                    score = (
                        self.eval_loss
                    )
                    if self.checkpoint.update(score):
                        self.logger.warning(
                            f"Saving new checkpoint at epoch {epoch} for train type "
                            f"{self.checkpoint.train_type} with {self.checkpoint.metric} "
                            f"of {self.checkpoint.best_score}"
                        )
                if end_epoch:
                    break

            if step == 2:
                self.main_backbone.head.set_weights(
                    self.detector_backbone_head.get_weights()
                )

        for step in range(int(self.checkpoint.trainer_step.numpy()), 4 + 1):
            if step == 1:
                message = "Training rpn"
            elif step == 2:
                message = "Training detector"
            elif step == 3:
                message = "Training rpn with fixed base"
            else:
                message = "Training detector with fixed base; same base as rpn"

            self.trainer_step.assign(step)
            _run_trainer_step(step, message)

    def train_rpn_fb_step(self, image: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Train the RPN using a fixed backbone network

        Args:
            image (Tensor): Input image, 3-D float32 Tensor of shape (H,W,3)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss of the rpn
        """
        x = tf.expand_dims(image, 0)
        H, W = x.shape[1:3]
        im_size = tf.constant([H, W])  # H, W

        with tf.GradientTape() as tape:
            # rpn loss
            rpnloss = self.rpn_forward_step(x, im_size, gt_bboxes)
            total_loss = rpnloss

        rpn_grads = tape.gradient(total_loss, self.rpn.trainable_variables)

        # clip gradients
        grad_clip = self.cfg["grad_clip"]
        rpn_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in rpn_grads]

        # apply grad
        self.rpn_optimizer.apply_gradients(zip(rpn_grads, self.rpn.trainable_variables))

        return total_loss

    def rpn_forward_step(
        self, image: Tensor, im_size: Tensor, gt_bboxes: Tensor
    ) -> Tensor:
        """Forward pass for the rpn step

        Args:
            image (Tensor): Input image, 4-D float32 Tensor of shape (1,H,W,3)
            im_size: size of image 1-D tensor of shape (2,)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        feat_map = self.main_backbone.head(image)
        rpn_deltas, rpn_scores = self.rpn(feat_map)
        anchors = generate_anchors(
            feat_map,
            tf.constant(self.cfg["rpn"]["anchor_base_size"]),
            tf.constant(self.cfg["rpn"]["stride"]),
            tf.constant(self.cfg["rpn"]["anchor_scales"]),
            tf.constant(self.cfg["rpn"]["anchor_ratios"]),
        )
        rpn_targets, rpn_labels = generate_rpn_targets(
            anchors,
            gt_bboxes,
            im_size,
            tf.constant(self.cfg["margin"]),
            tf.constant(self.cfg["clobber_positive"]),
            tf.constant(self.cfg["neg_iou_thresh"]),
            tf.constant(self.cfg["pos_iou_thresh"]),
            tf.constant(self.cfg["pos_anchors_perc"]),
            tf.constant(self.cfg["anchor_batch"]),
        )
        # rpn loss
        rpnloss = self.rpn_loss(rpn_targets, rpn_labels, rpn_deltas, rpn_scores)
        total_loss = rpnloss

    def train_rpn_step(self, image: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Train the RPN, including the backbone, for just one step

        Args:
            image (Tensor): Input image, 3-D float32 Tensor of shape (H,W,3)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        x = tf.expand_dims(image, 0)
        H, W = x.shape[1:3]
        im_size = tf.constant([H, W])  # H, W

        with tf.GradientTape() as tape:
            # rpn loss
            rpnloss = self.rpn_forward_step(x, im_size, gt_bboxes)

        rpn_grads, base_grads = tape.gradient(
            total_loss,
            [self.rpn.trainable_variables, self.main_backbone.head.trainable_variables],
        )
        # clip gradients
        grad_clip = self.cfg["grad_clip"]
        rpn_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in rpn_grads]
        base_grads = [tf.clip_by_value(g, -grad_clip, grad_clip) for g in base_grads]

        # apply grad
        self.rpn_optimizer.apply_gradients(zip(rpn_grads, self.rpn.trainable_variables))
        self.backbone_head_optimizer.apply_gradients(
            zip(base_grads, self.main_backbone.head.trainable_variables)
        )

        return total_loss

    def detector_forward_step(
        self, image: Tensor, im_size: Tensor, gt_bboxes: Tensor
    ) -> Tensor:
        """Forward pass for the detector step

        Args:
            image (Tensor): Input image, 4-D float32 Tensor of shape (1,H,W,3)
            im_size: size of image 1-D tensor of shape (2,)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        feat_map_det = self.detector_backbone_head(image)
        feat_map_rpn = self.main_backbone.head(image)
        anchors = generate_anchors(
            feat_map_det,
            tf.constant(self.cfg["rpn"]["anchor_base_size"], tf.int32),
            tf.constant(self.cfg["rpn"]["stride"], tf.int32),
            tf.constant(self.cfg["rpn"]["anchor_scales"], tf.float32),
            tf.constant(self.cfg["rpn"]["anchor_ratios"], tf.float32),
        )
        rpn_deltas, rpn_scores = self.rpn(feat_map_rpn)

        # decode proposals
        rpn_proposals = decode(anchors, rpn_deltas)

        # filter and suppress proposals
        rpn_scores = rpn_scores[:, 1]
        rpn_proposals, rpn_scores = filter_proposals(rpn_proposals, rpn_scores, im_size)

        rpn_proposals, rpn_scores = apply_nms(
            rpn_proposals,
            rpn_scores,
            tf.constant(self.cfg["rpn"]["nms_threshold"], tf.float32),
            tf.constant(self.cfg["rpn"]["top_n"], tf.int32),
        )

        # generate detector targets
        bbox_targets, bbox_labels = generate_detector_targets(
            rpn_proposals,
            gt_bboxes,
            tf.constant(self.cfg["bg_low"], tf.float32),
            tf.constant(self.cfg["bg_high"], tf.float32),
            tf.constant(self.cfg["fg_low"], tf.float32),
            tf.constant(self.cfg["pos_prop_perc"], tf.float32),
            tf.constant(self.cfg["prop_batch"], tf.int32),
        )

        # roi pooling
        rois = roi_pooling(
            feat_map_det,
            rpn_proposals,
            W,
            H,
            pool_size=tf.constant(self.cfg["pool_size"], tf.int32),
        )

        # process rois
        rois = self.main_backbone.tail(rois)

        # detector prediction
        bbox_deltas, cls_score = self.detector(rois)

        # calculate detector loss
        detectorloss = self.detector_loss(
            bbox_targets, bbox_labels, bbox_deltas, cls_score
        )
        return detectorloss

    def detector_forward_fb_step(
        self, image: Tensor, im_size: Tensor, gt_bboxes: Tensor
    ) -> Tensor:
        """Forward pass for the detector step

        Args:
            image (Tensor): Input image, 4-D float32 Tensor of shape (1,H,W,3)
            im_size: size of image 1-D tensor of shape (2,)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        feat_map = self.main_backbone.head(image)
        anchors = generate_anchors(
            feat_map_det,
            tf.constant(self.cfg["rpn"]["anchor_base_size"], tf.int32),
            tf.constant(self.cfg["rpn"]["stride"], tf.int32),
            tf.constant(self.cfg["rpn"]["anchor_scales"], tf.float32),
            tf.constant(self.cfg["rpn"]["anchor_ratios"], tf.float32),
        )
        rpn_deltas, rpn_scores = self.rpn(feat_map)

        # decode proposals
        rpn_proposals = decode(anchors, rpn_deltas)

        # filter and suppress proposals
        rpn_scores = rpn_scores[:, 1]
        rpn_proposals, rpn_scores = filter_proposals(rpn_proposals, rpn_scores, im_size)

        rpn_proposals, rpn_scores = apply_nms(
            rpn_proposals,
            rpn_scores,
            tf.constant(self.cfg["rpn"]["nms_threshold"], tf.float32),
            tf.constant(self.cfg["rpn"]["top_n"], tf.int32),
        )

        # generate detector targets
        bbox_targets, bbox_labels = generate_detector_targets(
            rpn_proposals,
            gt_bboxes,
            tf.constant(self.cfg["bg_low"], tf.float32),
            tf.constant(self.cfg["bg_high"], tf.float32),
            tf.constant(self.cfg["fg_low"], tf.float32),
            tf.constant(self.cfg["pos_prop_perc"], tf.float32),
            tf.constant(self.cfg["prop_batch"], tf.int32),
        )

        # roi pooling
        rois = roi_pooling(
            feat_map,
            rpn_proposals,
            W,
            H,
            pool_size=tf.constant(self.cfg["pool_size"], tf.int32),
        )

        # process rois
        rois = self.main_backbone.tail(rois)

        # detector prediction
        bbox_deltas, cls_score = self.detector(rois)

        # calculate detector loss
        detectorloss = self.detector_loss(
            bbox_targets, bbox_labels, bbox_deltas, cls_score
        )
        return detectorloss

    def train_detector_step(self, image: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Train the detector, including the backbone, for just one step

        Args:
            image (Tensor): Input image, 3-D float32 Tensor of shape (H,W,3)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        x = tf.expand_dims(image, 0)
        H, W = x.shape[1:3]  #
        im_size = tf.constant([H, W], tf.int32)

        with tf.GradientTape() as tape:
            # calculate detector loss
            detectorloss = self.detector_forward_step(x, im_size, gt_bboxes)

        detector_grads, base_grads, tail_grads = tape.gradient(
            detectorloss,
            [
                self.detector.trainable_variables,
                self.detector_backbone_head.trainable_variables,
                self.main_backbone.tail.trainable_variables,
            ],
        )

        # clip gradients
        grad_clip = self.cfg["grad_clip"]
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
            zip(tail_grads, self.main_backbone.tail.trainable_variables)
        )

        return detectorloss

    def train_detector_fb_step(self, image: Tensor, gt_bboxes: Tensor) -> Tensor:
        """Train the detector, with the backbone fixed, for just one step

        Args:
            image (Tensor): Input image, 3-D float32 Tensor of shape (H,W,3)
            gt_bboxes (Tensor): The ground truth bounding boxes, 2-D float32 Tensor of
             shape (num_boxes, 5)

        Returns:
            Tensor: The output loss, 0-D float32 Tensor, Scalar
        """
        x = tf.expand_dims(image, 0)
        H, W = x.shape[1:3]
        im_size = tf.constant([H, W], tf.int32)

        with tf.GradientTape() as tape:
            # calculate detector loss
            detectorloss = self.detector_forward_fb_step(x, im_size, gt_bboxes)

        detector_grads, tail_grads = tape.gradient(
            detectorloss,
            [
                self.detector.trainable_variables,
                self.main_backbone.tail.trainable_variables,
            ],
        )

        # clip gradients
        grad_clip = self.cfg["grad_clip"]
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

        return detectorloss
