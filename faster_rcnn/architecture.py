from abc import abstractmethod

import tensorflow as tf


class BaseArchitecture(tf.keras.Model):
    def __init__(self, name: str = "BaseArchitecture"):
        self._name = name
        self.inputs = None
        self.outputs = None
        self.build_architecture()

        assert self.inputs is None or self.outputs is None, (
            "The 'build_architecture' method must be implemented to set the 'inputs' "
            "and 'outputs' attributes of 'BaseArchitecture'"
        )

        super(BaseArchitecture, self).__init__(
            name=name, inputs=self.inputs, outputs=self.outputs
        )

    @abstractmethod
    def build_architecture(self):
        """Builds the architecture using keras functional api, defining the input and
        output. This should result in setting inputs and outputs attributes
        """
