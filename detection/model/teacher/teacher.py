from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.checkpoint import DetectionCheckpointer

from torch import nn

class Teacher(nn.Module):
    def __init__(self, backbone, proposal_generator, roi_heads):
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

def build_teacher(cfg):
    teacher_cfg = cfg.TEACHER
    backbone = build_backbone(teacher_cfg)
    if not 'Retina' in teacher_cfg.MODEL.META_ARCHITECTURE:
        proposal_generator = build_proposal_generator(teacher_cfg, backbone.output_shape())
        roi_heads = build_roi_heads(teacher_cfg, backbone.output_shape())
    else:
        proposal_generator = None
        roi_heads = None
    teacher = Teacher(backbone, proposal_generator, roi_heads)
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


