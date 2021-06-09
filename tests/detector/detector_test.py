import tensorflow as tf
from fasterrcnn.detection import Detector


class TestDetector:
    def test_detector(self, processed_roi, cfg):
        detector_cfg = cfg["detector"]
        detector = Detector(detector_cfg)

        assert detector.num_classes == detector_cfg["num_classes"]
        assert detector.input_channels == detector_cfg["input_channels"]

        bb, cl_score = detector(processed_roi)

        assert len(bb.shape) == 2 and len(cl_score.shape) == 2
        assert (
            bb.shape[-1] == detector_cfg["num_classes"] * 4
            and cl_score.shape[-1] == detector_cfg["num_classes"] + 1
        )
