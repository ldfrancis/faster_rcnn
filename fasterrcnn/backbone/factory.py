from .backbone import Backbone, BackboneHead, BackboneTail
from .constants import AVAILABLE_BACKBONES


def get_backbone(name):
    assert name in AVAILABLE_BACKBONES, f"{name} is not available"
    backbone_head = BackboneHead(name, name=f"{name}_BackboneHead")
    backbone_tail = BackboneTail(name, name=f"{name}_BackboneTail")
    backbone = Backbone(backbone_head, backbone_tail)
    return backbone, backbone_head, backbone_tail
