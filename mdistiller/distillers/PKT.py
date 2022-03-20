import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def pkt_loss(f_s, f_t, eps=1e-7):
    # Normalize each vector by its norm
    output_net_norm = torch.sqrt(torch.sum(f_s**2, dim=1, keepdim=True))
    f_s = f_s / (output_net_norm + eps)
    f_s[f_s != f_s] = 0
    target_net_norm = torch.sqrt(torch.sum(f_t**2, dim=1, keepdim=True))
    f_t = f_t / (target_net_norm + eps)
    f_t[f_t != f_t] = 0

    # Calculate the cosine similarity
    model_similarity = torch.mm(f_s, f_s.transpose(0, 1))
    target_similarity = torch.mm(f_t, f_t.transpose(0, 1))
    # Scale cosine similarity to 0..1
    model_similarity = (model_similarity + 1.0) / 2.0
    target_similarity = (target_similarity + 1.0) / 2.0
    # Transform them into probabilities
    model_similarity = model_similarity / torch.sum(
        model_similarity, dim=1, keepdim=True
    )
    target_similarity = target_similarity / torch.sum(
        target_similarity, dim=1, keepdim=True
    )
    # Calculate the KL-divergence
    loss = torch.mean(
        target_similarity
        * torch.log((target_similarity + eps) / (model_similarity + eps))
    )
    return loss


class PKT(Distiller):
    """
    Probabilistic Knowledge Transfer for deep representation learning
    Code from: https://github.com/passalis/probabilistic_kt
    """

    def __init__(self, student, teacher, cfg):
        super(PKT, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.PKT.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.PKT.LOSS.FEAT_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # lossess
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * pkt_loss(
            feature_student["pooled_feat"], feature_teacher["pooled_feat"]
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
