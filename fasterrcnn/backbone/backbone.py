from typing import Any, Dict, List, Union

import tensorflow as tf
from tensorflow import Tensor, Variable

from .constants import AVAILABLE_BACKBONES, RESNET101
from .extractor import Extractor, obtain_extractor_map
from .preprocessor import BasePreprocessor
from .tail_network import TailNetwork, obtain_tail_network_map


class BackboneHead(tf.keras.Model):
    def __init__(
        self,
        extractor: Union[str, Extractor] = "resnet101",
        preprocessor: BasePreprocessor = BasePreprocessor(),
        config: Dict[str, Any] = {},
        name: str = "BackboneHead",
    ) -> None:
        """Initialize the Head for the backbone CNN network for faster rcnn. This
        backbone head comprises of an extractor and a preprocessor. The preprocessor
        operates on the input image, normalizing it. While the extractor performs the
        task of feature extraction given the preprocessed input

        Args:
            extractor (str, optional): The CNN model to use in backbone for
            feature extraction. Defaults to "resnet101".
            preprocessor (BasePreprocessor): The preprocessor to use on an input image
             before passing into the extractor of the backbone
            config (Dict[str, Any], optional): Configuration settings for creating the
            backnone. Defaults to {}.

        Returns:
            None
        """
        self._name = name
        super(BackboneHead, self).__init__(name=name)

        if isinstance(extractor, str):
            assert (
                extractor in AVAILABLE_BACKBONES
            ), f"{extractor} is not available to be used in backbone"

            self.extractor = obtain_extractor_map()[extractor](f"{name}_Extractor")
        else:
            assert isinstance(
                extractor, Extractor
            ), f"extractor has to be an instance of Extractor, not {type(extractor)}"
            self.extractor = extractor

        self.preprocessor = preprocessor
        self.cfg = config

    def call(self, x: Tensor) -> Tensor:
        """Extract features from the input image

        Args:
            x (Tensor): The input image, 4-D float32 Tensor

        Returns:
            Tensor: Extracted feature map, 4-D float32 Tensor
        """
        x = self.preprocessor(x)
        x = self.extractor(x)

        return x


class BackboneTail(tf.keras.Model):
    def __init__(
        self,
        tail_network: Union[str, TailNetwork] = "resnet101",
        config: Dict[str, Any] = {},
        name: str = "BackboneHead",
    ) -> None:
        """Initialize the backbone CNN network for faster rcnn.

        Args:
            tail_network (Union[str, TailNetwork], optional): The CNN network to use on
             extracted image features in regions of interest. This further processes rois
             before they are fed to the detector. Defaults to "resnet101".
            config (Dict[str, Any], optional): Configuration settings for creating the
             BackboneTail. Defaults to {}.

        Returns:
            None
        """
        super(BackboneTail, self).__init__(name=name)

        if isinstance(tail_network, str):
            assert (
                tail_network in AVAILABLE_BACKBONES
            ), f"{tail_network} is not available to be used in backbone"

            self.tail_network = obtain_tail_network_map()[tail_network](
                f"{self._name}_TailNetwork"
            )
        else:
            assert isinstance(tail_network, TailNetwork), (
                "tail_network has to be an instance of TailNetwork, not "
                f"{type(tail_network)}"
            )
            self.tail_network = tail_network

        self.cfg = config

    def call(self, x: Tensor) -> Tensor:
        """Further process the pooled rois

        Args:
            x (Tensor): Input roi feature map. 4-D float32 Tensor

        Returns:
            Tensor: The processed roi features. 4-D float32 Tensor
        """
        x = self.tail_network(x)
        return x


class Backbone:
    def __init__(self, head: BackboneHead = None, tail: BackboneTail = None):
        """Initializes the backbone which consists of two parts; a head and a tail. The
        head helps with feature extraction from images while the tail is used on
        selected rois (regions of interest). The output from the tail is what is fed to
        the detector network for object class prediction and bounding box regression.

        Args:
            head (BackboneHead, optional): To be used for feature extraction. It has a
            preprocessor and an extractor that helps with this function. Defaults to
            None.
            tail (BackboneTail, optional): This takes extracted features in selected
            rois as input and returns a tensor that would be used by the detector for
            region classification and bounding box regression. Defaults to None.
        """
        self.head = head
        self.tail = tail
        assert self.head is not None
        assert isinstance(
            self.head, BackboneHead
        ), f"argument head must be an instance of BackboneHead not {type(head)}"
        assert self.tail is None or isinstance(self.tail, BackboneTail), (
            "supplied value for argument tail must be an instance of BackboneTail, not "
            f"{type(tail)}"
        )

    def __call__(self, x: Tensor, part: str = "head") -> Tensor:
        """Uses the backbone network to extract image features or process extracted
        region of interest (roi) features for the detector

        Args:
            x (Tensor): Input tensor for which to extract features or process for
            detection as depicted in the keword argument part
            part (str, optional): Determines whether the backbone head or tail would
            be applied to the input tensor, x. It must be either head or tail.
            Defaults to "head".

        Returns:
            Tensor: The tensor resulting as an output of whatever operation was applied
            on the input x based on the setting of the part argument
        """
        assert isinstance(
            part, str
        ), f"The argument, part, has to be a string not {type(part)}"
        assert part in [
            "head",
            "tail",
        ], f"Backbone's part has to be either head or tail"
        x = self.head(x) if part == "head" else self.tail(x)
        return x
