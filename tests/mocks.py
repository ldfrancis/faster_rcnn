import numpy as np
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import concat
from tensorflow.python.ops.variables import Variable


class MockDataset:
    def __init__(self):
        self.h, self.w = 678, 764
        self.limit = 10
        self.idx = 0

    @staticmethod
    def get_image():
        H, W = np.random.randint(600, 1000, (2,), dtype=np.int32)
        return tf.Variable(
            np.random.randint(0, 255, (H, W, 3)),
            dtype=tf.int32,
            name="mockimage",
        )

    def _get_bboxes(self):
        return tf.Variable(
            np.random.randint(0, 255, (4, 4)), dtype=tf.int32, name="mockimage"
        )

    def _get_lbl(self):
        lbl = tf.Variable([1, 1, 1, 1], dtype=tf.float32)
        return lbl

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):

        if self.idx >= 10:
            raise StopIteration

        # create example
        example = {
            "image": self.get_image(),
            "objects": {
                "box": self._get_bboxes(),
                "label": self._get_lbl(),
            },
        }

        self.idx += 1

        return example


class MockFeatureMap:
    @staticmethod
    def create_7x7_pooled_roi():
        return tf.Variable(
            np.random.randint(0, 255, (1, 7, 7, 1024)),
            dtype=tf.float32,
            name="mockimage",
        )

    @staticmethod
    def create_processed_roi():
        return tf.Variable(
            np.random.randint(0, 255, (1, 4, 4, 2048)),
            dtype=tf.float32,
            name="mockimage",
        )

    @staticmethod
    def create_featmap():
        H, W = np.random.randint(14, 36, (2,), dtype=np.int32)
        return tf.Variable(
            np.random.randint(0, 255, (1, H, W, 1024)),
            dtype=tf.float32,
            name="mockimage",
        )


mock_config = {
    "rpn": {
        "name": "rpn",
        "backbone": "resnet101",
        "anchor_base_size": 256,
        "anchor_ratios": [0.5, 1, 2],
        "anchor_scales": [0.125, 0.25, 0.5, 1, 2],
        "base_conv_channels": 512,
        "input_channels": 1024,
        "score_thresh": 0.7,
        "nms_threshold": 0.7,
        "top_n": 2000,
        "pool_size": 7,
    },
    "detector": {
        "name": "detector",
        "num_classes": 20,
        "input_channels": 2048,
        "top_n": 19,
        "score_thresh": 0.7,
        "nms_threshold": 0.7,
    },
    "trainer": {
        "name": "trainer",
        "base_size": 600,
        "stride": 16,
        "grad_clip": 10,
        "bg_low": 0,
        "bg_high": 0.3,
        "fg_low": 0.7,
        "pos_prop_perc": 0.5,
        "prop_batch": 256,
        "pool_size": 7,
        "margin": 100,
        "clobber_positive": False,
        "neg_iou_thresh": 0.3,
        "pos_iou_thresh": 0.7,
        "pos_anchors_perc": 0.5,
        "anchor_batch": 256,
        "epochs": 100,
        "backbone": "resnet101",
        "detector_lr": 1e-4,
        "backbone_head_lr": 1e-4,
        "backbone_tail_lr": 1e-4,
        "rpn_lr": 1e-4,
        "train_type": "4step",
    },
    "base_size": 600,
}
