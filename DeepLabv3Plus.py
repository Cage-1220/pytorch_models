# -*- coding:utf-8 -*-
# Author : lkq
# Data : 2019/4/1 14:20
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet50', 'resnet101']


model_urls = {
    #'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    #'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    #'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, dilation=dilation,
                     padding=1, bias=False)


class _ASPP(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, momentum, use_separable_conv=False):
        super(_ASPP, self).__init__()

        Conv = AtrousSeparableConvolution if (use_separable_conv and kernel_size > 1) else nn.Conv2d
        self.atrous_conv_bn_relu = nn.Sequential(
            Conv(inplanes, planes, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes, momentum=momentum),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.atrous_conv_bn_relu(x)

class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, momentum=0.1, use_separable_conv=False):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = _ASPP(inplanes, 256, 1, padding=0, dilation=dilations[0], momentum=momentum,
                           use_separable_conv=use_separable_conv)
        self.aspp2 = _ASPP(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], momentum=momentum,
                           use_separable_conv=use_separable_conv)
        self.aspp3 = _ASPP(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], momentum=momentum,
                           use_separable_conv=use_separable_conv)
        self.aspp4 = _ASPP(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], momentum=momentum,
                           use_separable_conv=use_separable_conv)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(256, momentum=momentum),
                                             nn.ReLU(inplace=True))

        self.reduce = nn.Sequential(
            nn.Conv2d(1280, 256, 1, bias=False),
            nn.BatchNorm2d(256, momentum=momentum),
            nn.ReLU(inplace=True)
        )

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.reduce(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, output_stride=16, momentum=0.1):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]  # dilated conv for last 3 blocks (9 layers)
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]  # 23+3 blocks (78 layers)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=momentum)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       momentum=momentum)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       momentum=momentum)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       momentum=momentum)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3],
                                       momentum=momentum)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=momentum),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation, momentum=momentum))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, momentum=momentum))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_channels=2048, momentum=0.1, use_separable_conv=False):
        super(Decoder, self).__init__()

        self.reduce_low_level = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48, momentum=momentum),
            nn.ReLU(inplace=True),
        )

        Conv = AtrousSeparableConvolution if use_separable_conv else nn.Conv2d
        self.last_conv = nn.Sequential(Conv(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256, momentum=momentum),
                                       nn.ReLU(inplace=True),
                                       Conv(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256, momentum=momentum),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_features):
        low_level_features = self.reduce_low_level(low_level_features)
        x = F.interpolate(x, size=low_level_features.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

    def forward(self, x):
        return self.body(x)


class DeepLabv3(nn.Module):
    """DeepLab v3+
    """

    def __init__(self, num_classes=21, backbone='resnet50', pretrained=True, output_stride=16, momentum=0.1,
                 use_separable_conv=False):
        super(DeepLabv3, self).__init__()
        if 'resnet50' in backbone:
            low_level_channels = 256
            features_channels = 2048
            self.backbone = resnet50(pretrained=pretrained)
        elif 'resnet101' in backbone:
            low_level_channels = 256
            features_channels = 2048
            self.backbone = resnet101( pretrained=pretrained)

        else:
            raise "[!] Backbone %s not supported yet!" % backbone

        self.aspp = ASPP(inplanes=features_channels, output_stride=output_stride, momentum=momentum,
                               use_separable_conv=use_separable_conv)
        self.decoder = Decoder(num_classes=num_classes, low_level_channels=low_level_channels, momentum=momentum,
                                     use_separable_conv=use_separable_conv)

    def forward(self, x):
        in_size = x.shape[2:]
        x, low_level_features = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        return F.interpolate(x, size=in_size, mode='bilinear', align_corners=False)


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model
