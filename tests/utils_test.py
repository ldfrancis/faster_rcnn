import math

import numpy as np
import pytest
import tensorflow as tf
from fasterrcnn.utils.bbox_utils import (
    bbox_overlap,
    decode,
    encode,
    swap_xy,
    to_center_width_height,
)
from fasterrcnn.utils.detector_targets_utils import generate_detector_targets
from fasterrcnn.utils.nms_utils import apply_nms, per_class_nms
from fasterrcnn.utils.proposals_utils import filter_proposals
from fasterrcnn.utils.roi_pooling_utils import normalize_bboxes, roi_pooling
from fasterrcnn.utils.rpn_anchors_utils import (
    generate_anchors,
    generate_reference_anchors,
)
from fasterrcnn.utils.rpn_targets_utils import generate_rpn_targets
from tensorflow.python.framework.tensor_shape import TensorShape


@pytest.fixture
def anchors():
    return tf.constant(
        np.array(
            [
                [30, 30, 286, 286],
                [200, 100, 456, 356],
                [-9, -5, 100, 251],
                [100, 10, 356, 266],
            ]
        ),
        dtype=tf.float32,
    )


@pytest.fixture
def gt_bboxes():
    return tf.constant(
        np.array(
            [
                [200, 100, 456, 356, 2],
                [100, 10, 356, 266, 3],
            ]
        ),
        dtype=tf.float32,
    )


@pytest.fixture
def zero_centered_bboxes():
    return tf.constant(
        [[-10, -10, 9, 9], [-100, -301, 99, 300], [-343, -57, 342, 56]],
        dtype=tf.float32,
    )


class TestUtils:
    @pytest.mark.parametrize(
        "scales_aspects",
        [([1, 1.5, 0.5], [1, 0.5, 1.5, 2]), ([1.5, 1, 2], [2, 0.5, 1.5])],
    )
    def test_generate_reference_anchors(self, scales_aspects):
        scales, aspects = scales_aspects
        reference_anchors = generate_reference_anchors(
            tf.constant(256), tf.constant(scales), tf.constant(aspects)
        )
        assert reference_anchors.shape == TensorShape([len(scales) * len(aspects), 4])

    @pytest.mark.parametrize(
        "scales_aspects",
        [([1, 1.5, 0.5], [1, 0.5, 1.5, 2]), ([1.5, 1, 2], [2, 0.5, 1.5])],
    )
    def test_generate_anchors(self, image_featmap, scales_aspects):
        scales, aspects = scales_aspects

        anchors = generate_anchors(
            image_featmap,
            tf.constant(256),
            tf.constant(16),
            tf.constant(scales),
            tf.constant(aspects),
        )
        assert anchors.shape == TensorShape(
            [tf.reduce_prod(image_featmap.shape[1:3]) * len(scales) * len(aspects), 4]
        )
        assert anchors.dtype == tf.float32
        assert math.floor(float(anchors[0, 0].numpy())) == math.floor(
            (-scales[0] * 256 * tf.math.sqrt(float(aspects[0])) / 2).numpy()
        )
        assert math.ceil(float(anchors[0, 2].numpy())) == math.ceil(
            (scales[0] * 256 * tf.math.sqrt(float(aspects[0])) / 2).numpy()
        )
        assert math.ceil(float(anchors[0, 1].numpy())) == math.ceil(
            (-scales[0] * 256 / tf.math.sqrt(float(aspects[0])) / 2).numpy()
        )
        assert math.floor(float(anchors[0, 3].numpy())) == math.floor(
            ((scales[0] * 256 / tf.math.sqrt(float(aspects[0]))) / 2).numpy()
        )

    def test_generate_rpn_targets(self, anchors, gt_bboxes):
        im_shape = tf.constant([600, 600], dtype=tf.int32)
        margin = tf.constant(10, dtype=tf.int32)
        targets, labels = generate_rpn_targets(
            anchors,
            gt_bboxes,
            im_shape,
            margin,
            clobber_positive=tf.constant(True),
            neg_iou_thresh=tf.constant(0.3),
            pos_iou_thresh=tf.constant(0.7),
            pos_anchors_perc=tf.constant(0.5),
            anchor_batch=tf.constant(2),
        )

        assert len(tf.where(labels == 1)[0]) == 1
        assert np.all(targets.numpy()[(1, 3), :] == 0)
        assert targets.shape == anchors.shape
        assert targets.dtype == tf.float32
        assert labels.dtype == tf.int32

    def test_generate_detector_targets(self, anchors, gt_bboxes):
        proposals = anchors
        targets, labels = generate_detector_targets(
            proposals,
            gt_bboxes,
            bg_low=tf.constant(0.0),
            bg_high=tf.constant(0.3),
            fg_low=tf.constant(0.7),
            pos_prop_perc=tf.constant(0.5),
            prop_batch=tf.constant(2),
        )
        
        assert len(tf.where(labels == 0)) == 1
        assert len(tf.where(labels > 0)) == 1
        assert len(tf.where(labels == -1)) == 2
        assert targets.shape == TensorShape([4, 4])
        assert targets.dtype == tf.float32
        assert labels.dtype == tf.int32

    def test_to_center_width_height(self, zero_centered_bboxes):

        cx, cy, w, h = to_center_width_height(zero_centered_bboxes)

        np.testing.assert_equal(cx, np.zeros_like(cx))
        np.testing.assert_equal(cy, np.zeros_like(cy))
        np.testing.assert_equal(w, np.array([[20], [200], [686]]))
        np.testing.assert_equal(h, np.array([[20], [602], [114]]))

    def test_encode(self, anchors):

        targets1 = encode(anchors, anchors)
        targets2 = encode(anchors, anchors + 1)

        np.testing.assert_equal(targets1, np.zeros_like(targets1))
        np.testing.assert_equal(targets2[:, 2:], np.zeros_like(targets2[:, 2:]))

        assert targets1.shape == anchors.shape
        assert targets2.shape == anchors.shape

    def test_decode(self, anchors):

        deltas = tf.zeros_like(anchors)
        gt_bboxes = decode(anchors, deltas)

        np.testing.assert_almost_equal(gt_bboxes.numpy(), anchors.numpy(), 0)

    def test_swap_xy(self, anchors):
        anchors1 = swap_xy(anchors)
        np.testing.assert_equal(anchors[:, 0].numpy(), anchors1[:, 1].numpy())
        np.testing.assert_equal(anchors[:, 1].numpy(), anchors1[:, 0].numpy())
        np.testing.assert_equal(anchors[:, 2].numpy(), anchors1[:, 3].numpy())
        np.testing.assert_equal(anchors[:, 3].numpy(), anchors1[:, 2].numpy())

    def test_bbox_overlap(self, anchors):
        overlap1 = bbox_overlap(anchors, anchors)
        diag = tf.linalg.diag_part(overlap1)

        np.testing.assert_almost_equal(diag, np.ones_like(diag))

    @pytest.mark.parametrize(
        "bboxes,scores",
        [
            (
                np.array(
                    [
                        [-10, -11, 11, 11],
                        [-10, -11, 10, 10],
                        [50, 100, 200, 300],
                        [0, 50, 50, 100],
                    ]
                ),
                np.array([0.9, 0.3, 0.5, 0.5]),
            )
        ],
    )
    def test_apply_nms(self, bboxes, scores):
        bboxes = tf.constant(bboxes, tf.float32)
        scores = tf.constant(scores, tf.float32)

        r_bboxes, r_scores = apply_nms(
            bboxes, scores, tf.constant(0.5), tf.constant(10)
        )
        assert r_bboxes.shape == TensorShape([3, 4])
        assert r_bboxes.shape[0] == r_scores.shape[0]
        np.testing.assert_equal(r_bboxes[0].numpy(), bboxes[0].numpy())
        np.testing.assert_equal(r_bboxes[1:].numpy(), bboxes[2:].numpy())

        r_bboxes, r_scores = apply_nms(bboxes, scores, tf.constant(0.5), tf.constant(1))
        assert r_bboxes.shape == TensorShape([1, 4])
        assert r_bboxes.shape[0] == r_scores.shape[0]
        np.testing.assert_equal(r_bboxes[0].numpy(), bboxes[0].numpy())

        r_bboxes, r_scores = apply_nms(bboxes, scores, tf.constant(0.5), tf.constant(2))
        assert r_bboxes.shape == TensorShape([2, 4])
        assert r_bboxes.shape[0] == r_scores.shape[0]
        np.testing.assert_equal(r_bboxes[0].numpy(), bboxes[0].numpy())

        r_bboxes, r_scores = apply_nms(
            bboxes, scores, tf.constant(1.0), tf.constant(10)
        )
        assert r_bboxes.shape == TensorShape([4, 4])

    @pytest.mark.parametrize(
        "bboxes, classes, scores1, scores2",
        [
            (
                np.array(
                    [
                        [-10, -11, 11, 11],
                        [-10, -11, 10, 10],
                        [50, 100, 200, 300],
                        [0, 50, 50, 100],
                    ]
                ),
                np.array([1, 1, 0, 0]),
                np.array([0.9, 0.3, 0.5, 0.5]),
                np.array([0.9, 0.7, 0.7, 0.7]),
            )
        ],
    )
    def test_per_class_nms(self, bboxes, classes, scores1, scores2):
        bboxes = tf.constant(bboxes, tf.float32)
        scores1 = tf.constant(scores1, tf.float32)
        scores2 = tf.constant(scores2, tf.float32)
        classes = tf.constant(classes, tf.int32)
        score_thresh = tf.constant(0.6)
        r_bboxes, r_scores = per_class_nms(
            bboxes,
            classes,
            scores2,
            tf.constant([600, 600]),
            score_thresh,
            tf.constant(0.7),
            tf.constant(9),
        )

        np.testing.assert_equal(
            r_bboxes.numpy(),
            np.array(
                [
                    [0.0, 0.0, 11.0, 11.0, 1.0],
                    [50.0, 100.0, 200.0, 300.0, 0.0],
                    [0.0, 50.0, 50.0, 100.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_equal(
            r_scores.numpy(),
            np.array(
                [0.9, 0.7, 0.7],
                dtype=np.float32,
            ),
        )

        r_bboxes, r_scores = per_class_nms(
            bboxes,
            classes,
            scores1,
            tf.constant([600, 600]),
            score_thresh,
            tf.constant(0.7),
            tf.constant(9),
        )

        np.testing.assert_equal(
            r_bboxes.numpy(),
            np.array(
                [
                    [0.0, 0.0, 11.0, 11.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_equal(
            r_scores.numpy(),
            np.array(
                [0.9],
                dtype=np.float32,
            ),
        )

    @pytest.mark.parametrize(
        "proposals, scores, expected_proposals1, expected_scores1, expected_proposals2, expected_scores2 ",
        [
            (
                np.array(
                    [
                        [-10, 20, 200, 300],
                        [20, 200, 100, 150],
                        [100, 200, 500, 600],
                        [-10, 45, 60, 128],
                    ]
                ),
                np.array([0.9, 0.8, 0.8, 0.3]),
                np.array(
                    [
                        [0.0, 20.0, 200.0, 300.0],
                        [100.0, 200.0, 499.0, 499.0],
                        [0.0, 45.0, 60.0, 128.0],
                    ],
                    dtype=np.float32,
                ),
                np.array(
                    [0.9, 0.8, 0.3],
                    dtype=np.float32,
                ),
                np.array(
                    [[0.0, 20.0, 200.0, 300.0], [100.0, 200.0, 499.0, 499.0]],
                    dtype=np.float32,
                ),
                np.array(
                    [0.9, 0.8],
                    dtype=np.float32,
                ),
            )
        ],
    )
    def test_filter_proposals(
        self,
        proposals,
        scores,
        expected_proposals1,
        expected_scores1,
        expected_proposals2,
        expected_scores2,
    ):
        proposals = tf.constant(proposals, tf.float32)
        scores = tf.constant(scores, tf.float32)
        score_thresh = tf.constant(0.3, tf.float32)
        inference = tf.constant(False, tf.bool)
        im_size = tf.constant([500, 500], tf.int32)
        _proposals, _scores = filter_proposals(
            proposals, scores, im_size, score_thresh, inference
        )

        np.testing.assert_equal(_proposals.numpy(), expected_proposals1)
        np.testing.assert_equal(_scores.numpy(), expected_scores1)

        score_thresh = tf.constant(0.7, tf.float32)
        inference = tf.constant(True, tf.bool)
        _proposals, _scores = filter_proposals(
            proposals, scores, im_size, score_thresh, inference
        )

        np.testing.assert_equal(_proposals.numpy(), expected_proposals2)
        np.testing.assert_equal(_scores.numpy(), expected_scores2)

    def test_normalize_bboxes(
        self,
        anchors,
    ):
        width = tf.constant(500)
        height = tf.constant(500)
        anchors = tf.constant(anchors, tf.float32)
        bboxes = normalize_bboxes(anchors, width, height)

        np.testing.assert_almost_equal(
            bboxes.numpy(),
            np.array(
                [
                    [0.06012, 0.06012, 0.573146, 0.573146],
                    [0.200401, 0.400802, 0.713427, 0.913828],
                    [0.0, 0.0, 0.503006, 0.200401],
                    [0.02004, 0.200401, 0.533066, 0.713427],
                ],
                dtype=np.float32,
            ),
            3,
        )

    def test_roi_pooling(self, image_featmap, anchors):
        width = tf.constant(500)
        height = tf.constant(500)
        rois = roi_pooling(image_featmap, anchors, width, height)

        assert rois.shape == TensorShape([4, 7, 7, 1024])
