import tensorflow as tf
from tensorflow import Tensor


class BasePreprocessor:
    def __init__(self, name: str = "BasePreprocessor") -> None:
        self._name = name

    def call(self, input_x: Tensor) -> Tensor:
        x = tf.keras.applications.imagenet_utils.preprocess_input(input_x)
        return x

    def __call__(self, input_x: Tensor) -> Tensor:
        return self.call(input_x)
