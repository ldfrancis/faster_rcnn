import tensorflow as tf

from ..architecture import BaseArchitecture


class DetectorArchitecture(BaseArchitecture):
    def __init__(
        self,
        input_channels: int = 1024,
        num_classes: int = 20,
        name: str = "DetectorArchitecture",
    ):
        """This builds the architecture to be used by the detector

        Args:
            input_channels (int, optional): Channels for the input Tensor to the
            detector. Defaults to 1024.
            num_classes (int, optional): The number of object classes to detect.
             Defaults to 20.
            name (str, optional): Name of the architecture. Defaults to
             "DetectorArchitecture".
        """
        self._name = name
        self._input_channels = input_channels
        self._num_classes = num_classes
        super(DetectorArchitecture, self).__init__(name=name)

    def build_architecture(self):
        input_x = tf.keras.Input(shape=(None, None, self._input_channels))
        x = tf.keras.layers.GlobalAveragePooling2D()(input_x)
        cls_x = tf.keras.layers.Dense(
            self._num_classes + 1, activation="softmax", name=f"{self._name}_classifier"
        )(x)
        bbox_pred = tf.keras.layers.Dense(
            self._num_classes * 4, name=f"{self._name}_bbox_pred"
        )(x)

        self.inputs = [input_x]
        self.outputs = [bbox_pred, cls_x]
