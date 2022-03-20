import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
import math

from ._base import Distiller


def feat_loss(source, target, margin):
    margin = margin.to(source)
    loss = (
        (source - margin) ** 2 * ((source > margin) & (target <= margin)).float()
        + (source - target) ** 2
        * ((source > target) & (target > margin) & (target <= 0)).float()
        + (source - target) ** 2 * (target > 0).float()
    )
    return torch.abs(loss).mean(dim=0).sum()


class ConnectorConvBN(nn.Module):
    def __init__(self, s_channels, t_channels, kernel_size=1):
        super(ConnectorConvBN, self).__init__()
        self.s_channels = s_channels
        self.t_channels = t_channels
        self.connectors = nn.ModuleList(
            self._make_conenctors(s_channels, t_channels, kernel_size)
        )

    def _make_conenctors(self, s_channels, t_channels, kernel_size):
        assert len(s_channels) == len(t_channels), "unequal length of feat list"
        connectors = nn.ModuleList(
            [
                self._build_feature_connector(t, s, kernel_size)
                for t, s in zip(t_channels, s_channels)
            ]
        )
        return connectors

    def _build_feature_connector(self, t_channel, s_channel, kernel_size):
        C = [
            nn.Conv2d(
                s_channel,
                t_channel,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.BatchNorm2d(t_channel),
        ]
        for m in C:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return nn.Sequential(*C)

    def forward(self, g_s):
        out = []
        for i in range(len(g_s)):
            out.append(self.connectors[i](g_s[i]))

        return out


class OFD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(OFD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.OFD.LOSS.CE_WEIGHT
        self.feat_loss_weight = cfg.OFD.LOSS.FEAT_WEIGHT
        self.init_ofd_modules(
            tea_channels=self.teacher.get_stage_channels()[1:],
            stu_channels=self.student.get_stage_channels()[1:],
            bn_before_relu=self.teacher.get_bn_before_relu(),
            kernel_size=cfg.OFD.CONNECTOR.KERNEL_SIZE,
        )

    def init_ofd_modules(
        self, tea_channels, stu_channels, bn_before_relu, kernel_size=1
    ):
        tea_channels, stu_channels = self._align_list(tea_channels, stu_channels)
        self.connectors = ConnectorConvBN(
            stu_channels, tea_channels, kernel_size=kernel_size
        )

        self.margins = []
        for idx, bn in enumerate(bn_before_relu):
            margin = []
            std = bn.weight.data
            mean = bn.bias.data
            for (s, m) in zip(std, mean):
                s = abs(s.item())
                m = m.item()
                if norm.cdf(-m / s) > 0.001:
                    margin.append(
                        -s
                        * math.exp(-((m / s) ** 2) / 2)
                        / math.sqrt(2 * math.pi)
                        / norm.cdf(-m / s)
                        + m
                    )
                else:
                    margin.append(-3 * s)
            margin = torch.FloatTensor(margin).to(std.device)
            self.margins.append(margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.connectors.parameters())

    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def get_extra_parameters(self):
        num_p = 0
        for p in self.connectors.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            _, feature_teacher = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_feat = self.feat_loss_weight * self.ofd_loss(
            feature_student["preact_feats"][1:], feature_teacher["preact_feats"][1:]
        )
        losses_dict = {"loss_ce": loss_ce, "loss_kd": loss_feat}
        return logits_student, losses_dict

    def ofd_loss(self, feature_student, feature_teacher):
        feature_student, feature_teacher = self._align_list(
            feature_student, feature_teacher
        )
        feature_student = [
            self.connectors.connectors[idx](feat)
            for idx, feat in enumerate(feature_student)
        ]

        loss_distill = 0
        feat_num = len(feature_student)
        for i in range(feat_num):
            loss_distill = loss_distill + feat_loss(
                feature_student[i],
                F.adaptive_avg_pool2d(
                    feature_teacher[i], feature_student[i].shape[-2:]
                ).detach(),
                self.margins[i],
            ) / 2 ** (feat_num - i - 1)
        return loss_distill

    def _align_list(self, *input_list):
        min_len = min([len(l) for l in input_list])
        return [l[-min_len:] for l in input_list]
