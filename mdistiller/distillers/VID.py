import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ._base import Distiller
from ._common import get_feat_shapes


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, padding=0, bias=False, stride=stride
    )


def vid_loss(regressor, log_scale, f_s, f_t, eps=1e-5):
    # pool for dimentsion match
    s_H, t_H = f_s.shape[2], f_t.shape[2]
    if s_H > t_H:
        f_s = F.adaptive_avg_pool2d(f_s, (t_H, t_H))
    elif s_H < t_H:
        f_t = F.adaptive_avg_pool2d(f_t, (s_H, s_H))
    else:
        pass
    pred_mean = regressor(f_s)
    pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
    pred_var = pred_var.view(1, -1, 1, 1).to(pred_mean)
    neg_log_prob = 0.5 * ((pred_mean - f_t) ** 2 / pred_var + torch.log(pred_var))
    loss = torch.mean(neg_log_prob)
    return loss


class VID(Distiller):
    """
    Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation
    """

    def __init__(self, student, teacher, cfg):
        super(VID, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.VID.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.VID.LOSS.FEAT_WEIGHT
        self.init_pred_var = cfg.VID.INIT_PRED_VAR
        self.eps = cfg.VID.EPS
        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.VID.INPUT_SIZE
        )
        feat_s_channels = [s[1] for s in feat_s_shapes[1:]]
        feat_t_channels = [s[1] for s in feat_t_shapes[1:]]
        self.init_vid_modules(feat_s_channels, feat_t_channels)

    def init_vid_modules(self, feat_s_shapes, feat_t_shapes):
        self.regressors = nn.ModuleList()
        self.log_scales = []
        for s, t in zip(feat_s_shapes, feat_t_shapes):
            regressor = nn.Sequential(
                conv1x1(s, t), nn.ReLU(), conv1x1(t, t), nn.ReLU(), conv1x1(t, t)
            )
            self.regressors.append(regressor)
            log_scale = torch.nn.Parameter(
                np.log(np.exp(self.init_pred_var - self.eps) - 1.0) * torch.ones(t)
            )
            self.log_scales.append(log_scale)

    def get_learnable_parameters(self):
        parameters = super().get_learnable_parameters()
        for regressor in self.regressors:
            parameters += list(regressor.parameters())
        return parameters

    def get_extra_parameters(self):
        num_p = 0
        for regressor in self.regressors:
            for p in regressor.parameters():
                num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_vid = 0
        for i in range(len(feature_student["feats"][1:])):
            loss_vid += vid_loss(
                self.regressors[i],
                self.log_scales[i],
                feature_student["feats"][1:][i],
                feature_teacher["feats"][1:][i],
                self.eps,
            )
        loss_vid = self.feat_loss_weight * loss_vid
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_vid,
        }
        return logits_student, losses_dict
