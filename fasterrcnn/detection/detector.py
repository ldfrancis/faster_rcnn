from typing import Any, Dict, Tuple

import tensorflow as tf
from tensorflow import Tensor

from .architecture import DetectorArchitecture


class Detector(tf.keras.Model):
    def __init__(
        self,
        detector_config: Dict[str, Any] = {},
    ) -> None:
        """The detection network for faster rcnn

        Args:
            detector_config (Dict[str, Any], optional): config dict for building the
             detection network. Defaults to {}.
        """
        self._name = detector_config["name"]
        super().__init__(name=self._name)
        self._cfg = detector_config
        self.num_classes = self._cfg["num_classes"]
        self.input_channels = self._cfg["input_channels"]

        self._architecture = DetectorArchitecture(
            self.input_channels, self.num_classes, name=f"{self._name}_Architecture"
        )

        self.build((1, 4, 4, self.input_channels))

    def call(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Obtain bounding box deltas/offset and corresponding class scores given a
        processed region of interest from the feature map extracted by the backbone
        network

        Args:
            x (Tensor): A processed region of interest, output from the backbone tail.
             4-D float32 Tensor of shape (batch, 4, 4, num_input_channels)

        Returns:
            bbox_deltas, cls_scores (Tuple[Tensor, Tensor]): The bounding box
             deltas/offsets and corresponding class scores. bbox_deltas is of shape
             (num_rois, 4*num_classes) while cls_scores is of shape
             (num_rois, num_classes+1)
        """
        bbox_deltas, cls_scores = self._architecture(x)

        return bbox_deltas, cls_scores
