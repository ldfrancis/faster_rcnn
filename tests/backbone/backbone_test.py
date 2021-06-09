import pytest
import tensorflow as tf
from fasterrcnn.backbone.backbone import Backbone, BackboneHead, BackboneTail
from fasterrcnn.backbone.constants import AVAILABLE_BACKBONES
from fasterrcnn.backbone.extractor import Extractor
from fasterrcnn.backbone.preprocessor import BasePreprocessor
from fasterrcnn.backbone.tail_network import TailNetwork


@pytest.mark.parametrize("backbone", AVAILABLE_BACKBONES)
class TestBackbone:
    def test_backbonehead(
        self,
        backbone,
        cfg,
        image,
    ):
        backbonehead = BackboneHead(backbone, config=cfg)
        assert isinstance(backbonehead.extractor, Extractor)
        assert isinstance(backbonehead.preprocessor, BasePreprocessor)
        assert isinstance(backbonehead.cfg, dict)

        image_ext = backbonehead(tf.expand_dims(image, 0))

        assert len(image_ext.shape) == 4
        assert image_ext.dtype == tf.float32
        assert image_ext.shape[-1] == 1024

    def test_backbonetail(
        self,
        backbone,
        cfg,
        pooled_roi_7x7,
    ):
        backbonetail = BackboneTail(backbone, config=cfg)
        assert isinstance(backbonetail.tail_network, TailNetwork)
        assert isinstance(backbonetail.cfg, dict)

        x = backbonetail(pooled_roi_7x7)

        assert len(x.shape) == 4
        assert x.dtype == tf.float32
        assert x.shape[-1] == 2048

    def test_backbone(
        self,
        backbone,
        image,
        cfg,
        pooled_roi_7x7,
    ):
        backbonetail = BackboneTail(backbone, config=cfg)
        backbonehead = BackboneHead(backbone, config=cfg)
        backbone = Backbone(backbonehead, backbonetail)

        assert isinstance(backbone.tail, BackboneTail)
        assert isinstance(backbone.head, BackboneHead)

        featmap = backbone.head(tf.expand_dims(image, 0))
        x = backbone.tail(pooled_roi_7x7)

        assert len(featmap.shape) == 4
        assert featmap.dtype == tf.float32
        assert featmap.shape[-1] == 1024

        assert len(x.shape) == 4
        assert x.dtype == tf.float32
        assert x.shape[-1] == 2048
