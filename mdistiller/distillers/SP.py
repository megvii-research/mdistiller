import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def sp_loss(g_s, g_t):
    return sum([similarity_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)])


def similarity_loss(f_s, f_t):
    bsz = f_s.shape[0]
    f_s = f_s.view(bsz, -1)
    f_t = f_t.view(bsz, -1)

    G_s = torch.mm(f_s, torch.t(f_s))
    G_s = torch.nn.functional.normalize(G_s)
    G_t = torch.mm(f_t, torch.t(f_t))
    G_t = torch.nn.functional.normalize(G_t)

    G_diff = G_t - G_s
    loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
    return loss


class SP(Distiller):
    """Similarity-Preserving Knowledge Distillation, ICCV2019"""

    def __init__(self, student, teacher, cfg):
        super(SP, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.SP.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.SP.LOSS.FEAT_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * sp_loss(
            [feature_student["feats"][-1]], [feature_teacher["feats"][-1]]
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
