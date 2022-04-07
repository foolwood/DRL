# Copyright (C) Alibaba Group Holding Limited. 

# Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

# BSD 3-Clause License
#
# Copyright (c) Soumith Chintala 2016,
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import torch.nn as nn


__all__ = ['ResNet2d', 'resnet10_2d', 'resnet18_2d', 'resnet26_2d', 'resnet34_2d',
           'resnet50_2d', 'resnet101_2d', 'resnet152_2d', 'resnet200_2d',
           'resnext50_32x4d_2d', 'resnext101_32x8d_2d', 'wide_resnet50_2_2d',
           'wide_resnet101_2_2d']


def conv1x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, dilation, dilation),
                     groups=groups,
                     bias=False,
                     dilation=(1, dilation, dilation))


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=(1, stride, stride),
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv1x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv1x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet2d(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 dropout=0.5, inplanes=3, first_stride=2, norm_layer=None, last_pool=True):
        super(ResNet2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if not last_pool and num_classes is not None:
            raise ValueError('num_classes should be None when last_pool=False')
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(inplanes, self.inplanes, kernel_size=(1, 7, 7),
                               stride=(1, first_stride, first_stride),
                               padding=(0, 3, 3), bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1)) if last_pool else None
        if num_classes is None:
            self.dropout = None
            self.fc = None
        else:
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.out_planes = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avgpool:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            if self.dropout and self.fc:
                x = self.dropout(x)
                x = self.fc(x)

        return x


def resnet10_2d(**kwargs):
    return ResNet2d(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet18_2d(**kwargs):
    return ResNet2d(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet26_2d(**kwargs):
    return ResNet2d(Bottleneck, [2, 2, 2, 2], **kwargs)


def resnet34_2d(**kwargs):
    return ResNet2d(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_2d(**kwargs):
    return ResNet2d(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_2d(**kwargs):
    return ResNet2d(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_2d(**kwargs):
    return ResNet2d(Bottleneck, [3, 8, 36, 3], **kwargs)


def resnet200_2d(**kwargs):
    return ResNet2d(Bottleneck, [3, 24, 36, 3], **kwargs)


def resnext50_32x4d_2d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return ResNet2d(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101_32x8d_2d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return ResNet2d(Bottleneck, [3, 4, 23, 3], **kwargs)


def wide_resnet50_2_2d(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return ResNet2d(Bottleneck, [3, 4, 6, 3], **kwargs)


def wide_resnet101_2_2d(**kwargs):
    kwargs['width_per_group'] = 64 * 2
    return ResNet2d(Bottleneck, [3, 4, 23, 3], **kwargs)
