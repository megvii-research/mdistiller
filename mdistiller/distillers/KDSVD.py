import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kdsvd_loss(g_s, g_t, k):
    v_sb = None
    v_tb = None
    losses = []
    for i, f_s, f_t in zip(range(len(g_s)), g_s, g_t):
        u_t, s_t, v_t = svd(f_t, k)
        u_s, s_s, v_s = svd(f_s, k + 3)
        v_s, v_t = align_rsv(v_s, v_t)
        s_t = s_t.unsqueeze(1)
        v_t = v_t * s_t
        v_s = v_s * s_t

        if i > 0:
            s_rbf = torch.exp(-(v_s.unsqueeze(2) - v_sb.unsqueeze(1)).pow(2) / 8)
            t_rbf = torch.exp(-(v_t.unsqueeze(2) - v_tb.unsqueeze(1)).pow(2) / 8)

            l2loss = (s_rbf - t_rbf.detach()).pow(2)
            l2loss = torch.where(
                torch.isfinite(l2loss), l2loss, torch.zeros_like(l2loss)
            )
            losses.append(l2loss.sum())

        v_tb = v_t
        v_sb = v_s

    bsz = g_s[0].shape[0]
    losses = [l / bsz for l in losses]
    return sum(losses)


def svd(feat, n=1):
    size = feat.shape
    assert len(size) == 4

    x = feat.view(size[0], size[1] * size[2], size[3]).float()
    u, s, v = torch.svd(x)

    u = removenan(u)
    s = removenan(s)
    v = removenan(v)

    if n > 0:
        u = F.normalize(u[:, :, :n], dim=1)
        s = F.normalize(s[:, :n], dim=1)
        v = F.normalize(v[:, :, :n], dim=1)

    return u, s, v


def removenan(x):
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x


def align_rsv(a, b):
    cosine = torch.matmul(a.transpose(-2, -1), b)
    max_abs_cosine, _ = torch.max(torch.abs(cosine), 1, keepdim=True)
    mask = torch.where(
        torch.eq(max_abs_cosine, torch.abs(cosine)),
        torch.sign(cosine),
        torch.zeros_like(cosine),
    )
    a = torch.matmul(a, mask)
    return a, b


class KDSVD(Distiller):
    """
    Self-supervised Knowledge Distillation using Singular Value Decomposition
    original Tensorflow code: https://github.com/sseung0703/SSKD_SVD
    """

    def __init__(self, student, teacher, cfg):
        super(KDSVD, self).__init__(student, teacher)
        self.k = cfg.KDSVD.K
        self.ce_loss_weight = cfg.KDSVD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.KDSVD.LOSS.FEAT_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * kdsvd_loss(
            feature_student["feats"][1:], feature_teacher["feats"][1:], self.k
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_feat,
        }
        return logits_student, losses_dict
