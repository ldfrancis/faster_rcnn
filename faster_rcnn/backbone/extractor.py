from typing import Dict, Type

import tensorflow as tf

from ..architecture import BaseArchitecture
from .constants import RESNET101


class Extractor(BaseArchitecture):
    def __init__(self, name: str = "Extractor"):
        super(Extractor, self).__init__(name=name)


class ResNet101Extractor(Extractor):
    def __init__(self, name: str = "ResNet101Extractor"):
        super(ResNet101Extractor, self).__init__(name=name)

    def build_architecture(self):
        resnet101_base = tf.keras.applications.ResNet101(
            input_shape=(None, None, 3), include_top=False
        )
        output_layer_name = "conv4_block23_out"
        self.inputs = [resnet101_base.input]
        self.outputs = [resnet101_base.get_layer(output_layer_name).output]


def obtain_extractor_map() -> Dict[str, Type[Extractor]]:
    """Create a dictionary that maps backbone string names to extractor classes

    Returns:
        Dict[str, Type[Extractor]]: A dictionary for extractors
    """
    extr_map = {RESNET101: ResNet101Extractor}

    return extr_map
