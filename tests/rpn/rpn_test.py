from fasterrcnn.rpn.rpn import RPN


class TestRPN:
    def test_rpn(self, image_featmap, cfg):
        rpn_config = cfg["rpn"]
        rpn = RPN(rpn_config)

        _, H, W, _ = image_featmap.shape

        bb, bb_scores = rpn(image_featmap)

        assert bb.shape[-1] == 4 and bb_scores.shape[-1] == 2
        assert len(bb.shape) == 2 and len(bb_scores.shape) == 2

        assert bb.shape[0] == H * W * len(rpn_config["anchor_ratios"]) * len(
            rpn_config["anchor_scales"]
        ) and bb_scores.shape[0] == H * W * len(rpn_config["anchor_ratios"]) * len(
            rpn_config["anchor_scales"]
        )
