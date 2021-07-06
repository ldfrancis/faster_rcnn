from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow.python.framework.ops import Tensor


class RPN(tf.keras.Model):
    def __init__(self, rpn_config: Dict[str, Any] = {}):
        """Create region proposal network instance

        Args:
            rpn_config (Dict[str, Any], optional): Config dict for building the rpn.
             Defaults to {}.
        """
        self._cfg = rpn_config
        self._name = rpn_config["name"]
        super(RPN, self).__init__(name=self._name)

        self.anchor_bs = self._cfg["anchor_base_size"]
        self.anchor_ratios = self._cfg["anchor_ratios"]
        self.anchor_scales = self._cfg["anchor_scales"]
        self.base_conv_channels = self._cfg["base_conv_channels"]
        self.input_channels = self._cfg["input_channels"]

        # number of anchors
        k = len(self.anchor_ratios) * len(self.anchor_scales)

        self.base_conv = tf.keras.layers.Conv2D(
            self.base_conv_channels, 3, 1, "same", name=f"{self._name}_base_conv"
        )
        self.bbox_delta = tf.keras.layers.Conv2D(
            4 * k, 1, 1, "same", name=f"{self._name}_bbox_conv"
        )
        self.objectness = tf.keras.layers.Conv2D(
            2 * k, 1, 1, "same", name=f"{self._name}_cls_conv"
        )

        self.build((1, None, None, self.input_channels))

    def call(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Propose object bounding boxes for an input feature map

        Args:
            x (Tensor): Input feature map. 4-D float32 Tensor

        Returns:
            Tuple[Tensor,Tensor]: deltas/offsets for proposed bounding boxes and their
             respective objectness scores
        """
        h = self.base_conv(x)
        bbox_deltas = tf.reshape(self.bbox_delta(h), (-1, 4))
        bbox_prob = tf.reshape(self.objectness(h), (-1, 2))
        bbox_prob = tf.nn.softmax(bbox_prob, axis=-1)

        return bbox_deltas, bbox_prob
