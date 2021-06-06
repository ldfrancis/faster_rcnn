from typing import Any, Dict

import tensorflow as tf

from ..utils.config_utils import baseDetectorConfig
from .architecture import DetectorArchitecture


class Detector(tf.keras.Model):
    def __init__(
        self,
        detector_config: Dict[str, Any] = baseDetectorConfig,
    ) -> None:
        self.name = detector_config["name"]
        super().__init__(name=self.name)
        self._cfg = detector_config
        self.num_classes = self._cfg["num_classes"]
        self.input_channels = self._cfg["input_channels"]
        self.name = self._cfg["name"]

        self._architecture = DetectorArchitecture(
            self.input_channels, self.num_classes, name=self.name
        )

        self.build((1, 7, 7, self.input_channels))

    def call(self, x):
        x = self._architecture(x)
