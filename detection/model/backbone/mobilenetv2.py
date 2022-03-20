"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone, FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from detectron2.layers import (
    Conv2d,
    DeformConv,
    FrozenBatchNorm2d,
    ModulatedDeformConv,
    ShapeSpec,
    get_norm,
)

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride, bn):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        get_norm(bn, oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        get_norm(bn, oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bn):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                get_norm(bn, hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                get_norm(bn, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                get_norm(bn, hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                get_norm(bn, hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                get_norm(bn, oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self



class MobileNetV2(Backbone):
    def __init__(self, cfg, input_shape, width_mult = 1.):
        super(MobileNetV2, self).__init__()
        self._out_features        = cfg.MODEL.MOBILENETV2.OUT_FEATURES
        bn = cfg.MODEL.MOBILENETV2.NORM
        freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
        

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1, ''],
            [6,  24, 2, 2, 'm2'],
            [6,  32, 3, 2, 'm3'],
            [6,  64, 4, 2, ''],
            [6,  96, 3, 1, 'm4'],
            [6, 160, 3, 2, ''],
            [6, 320, 1, 1, 'm5'],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(input_shape.channels, input_channel, 2, bn)]
        if freeze_at >= 1:
            for p in layers[0].parameters():
                p.requires_grad = False
            layers[0] = FrozenBatchNorm2d.convert_frozen_batchnorm(layers[0])
        # building inverted residual blocks
        block = InvertedResidual
        self.stage_name = ['']
        self._out_feature_channels = {}
        self._out_feature_strides = {}
        cur_stride = 2
        cur_stage = 2
        for t, c, n, s, name in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            cur_stride = cur_stride * s
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, bn))
                if cur_stage <= freeze_at :
                    layers[-1].freeze()
                if name != '' and i == n-1:
                    self._out_feature_channels[name] = output_channel
                    self._out_feature_strides[name] = cur_stride
                    cur_stage += 1
                input_channel = output_channel
                self.stage_name.append(name if i == n-1 else '')
        self.features = nn.Sequential(*layers)
        # building last several layers
#        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
#        self.conv = conv_1x1_bn(input_channel, output_channel)
#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        output = {}
        for i in range(len(self.features)):
            x = self.features[i](x)
            if self.stage_name[i] in self._out_features:
                output[self.stage_name[i]] = x
        return output
        '''
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        '''

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


        
@BACKBONE_REGISTRY.register()
def build_mobilenetv2_backbone(cfg, input_shape):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(cfg, input_shape)



