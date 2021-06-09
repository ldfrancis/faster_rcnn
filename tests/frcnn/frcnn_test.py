from fasterrcnn.backbone.backbone import Backbone
from fasterrcnn.detection.detector import Detector
from fasterrcnn.frcnn import FRCNN
from fasterrcnn.rpn.rpn import RPN


class TestFRCNN:
    def test_frcnn(self, image, cfg):
        frcnn = FRCNN(cfg)

        assert isinstance(frcnn.backbone, Backbone)
        assert isinstance(frcnn.rpn, RPN)
        assert isinstance(frcnn.detector, Detector)

        s_bb, s_scores = frcnn(image)

        assert s_bb.shape[0] == s_scores.shape[0]
        assert s_bb.shape[-1] == 5
