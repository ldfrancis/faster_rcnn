from abc import abstractmethod

import tensorflow as tf


class BaseArchitecture(tf.keras.Model):
    def __init__(self, name: str = "BaseArchitecture"):
        """Base architecture building class. Helps when using the keras functional api
        to build model architectures. In the build_architecture method, the
        architecture is built with the funtional api and the inputs and outputs are
        assigned to self.inputs and self.outputs

        Args:
            name (str, optional): [description]. Defaults to "BaseArchitecture".
        """
        self._name = name
        self.inputs = None
        self.outputs = None
        self.build_architecture()

        assert self.inputs is not None and self.outputs is not None, (
            "The 'build_architecture' method must be implemented to set the 'inputs' "
            "and 'outputs' attributes of 'BaseArchitecture'"
        )

        super(BaseArchitecture, self).__init__(
            name=name, inputs=self.inputs, outputs=self.outputs
        )

    @abstractmethod
    def build_architecture(self):
        """Builds the architecture using keras functional api, defining the input and
        output. This should result in setting inputs and outputs attributes. self.inputs
        and self.outputs must be assigned and not be None after this method is called.
        """
