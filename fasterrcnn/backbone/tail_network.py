from typing import Dict, Type

import tensorflow as tf

from ..architecture import BaseArchitecture
from .constants import RESNET101


class TailNetwork(BaseArchitecture):
    """The architecture for the tail network of the backbone"""

    def __init__(self, name: str = "TailNetwork"):
        super(TailNetwork, self).__init__(name=name)


class ResNet101TailNetwork(TailNetwork):
    """Tail network from resnet101"""

    def __init__(self, name: str = "ResNet101TailNetwork"):
        super(ResNet101TailNetwork, self).__init__(name=name)

    def get_layer_output_tensor(self, input_layer_name, layer_names, tensors, input_x):
        try:
            return tensors[layer_names.index(input_layer_name)]
        except:
            return input_x

    def get_tensor_name(self, tensor):
        name = tensor.name.split("/")[0]
        # remove tensor number
        name_blocks = name.split("_")
        try:
            if type(int(name_blocks[-1])) == type(1):
                name_blocks = name_blocks[:-1]
                name = "_".join(name_blocks)
        except:
            name = "_".join(name_blocks)

        return name

    def build_architecture(self):
        base_arch = tf.keras.applications.ResNet101()
        output_layer_name = "conv5_block3_out"
        all_layers = base_arch.layers

        tensors = []
        layer_names = []

        input_x = tf.keras.layers.Input(shape=(7, 7, 1024), name="conv4_block23_out")
        layer = tf.keras.layers.Conv2D(
            512, (1, 1), (2, 2), "same", name="conv5_block1_1_conv"
        )
        layer.build((1, 7, 7, 1024))
        layer.set_weights(base_arch.get_layer("conv5_block1_1_conv").get_weights())
        x = layer(input_x)
        tensors += [input_x, x]
        layer_names += ["conv4_block23_out", "conv5_block1_1_conv"]

        start_layer_ind = (
            all_layers.index(base_arch.get_layer("conv5_block1_1_conv")) + 1
        )
        end_layer_ind = all_layers.index(base_arch.get_layer(output_layer_name))

        for layer in all_layers[start_layer_ind : end_layer_ind + 1]:
            inputs = layer.input
            if isinstance(inputs, list):
                inputs = [
                    self.get_layer_output_tensor(
                        self.get_tensor_name(inp), layer_names, tensors, x
                    )
                    for inp in inputs
                ]
            else:
                inputs = self.get_layer_output_tensor(
                    self.get_tensor_name(inputs), layer_names, tensors, x
                )
            if layer.name == "conv5_block1_0_conv":
                layer = tf.keras.layers.Conv2D(
                    2048, (1, 1), (2, 2), "same", name="conv5_block1_0_conv"
                )
                layer.build((1, 7, 7, 1024))
                layer.set_weights(
                    base_arch.get_layer("conv5_block1_0_conv").get_weights()
                )
            x = layer(inputs)
            tensors += [x]
            layer_names += [layer.name]

        self.inputs = [input_x]
        self.outputs = [x]


def obtain_tail_network_map() -> Dict[str, Type[TailNetwork]]:
    """Create a dictionary that maps backbone string names to TailNetwork classes

    Returns:
        Dict[str, BaseArchitecture]: A dictionary for extractors
    """
    tn_map = {RESNET101: ResNet101TailNetwork}

    return tn_map
