# -*- coding:utf-8 -*-
# Author : lkq
# Data : 2019/4/12 14:23
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 kernel
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


# get BasicBlock which layers < 50(18, 34)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.BN = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes) # outplane is not in_planes*self.expansion, is planes
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x   # mark the data before BasicBlock
        x = self.conv1(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN(x)  # BN operation is before relu operation
        if self.downsample is not None:  # is not None
            residual = self.downsample(residual)  # resize the channel
        x += residual
        x = self.relu(x)
        return x


class Bottleneck(nn.Module):
    expansion = 4 # the factor of the last layer of BottleBlock and the first layer of it

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes,stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels //4,3,padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv == True:
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4,in_channels // 4,3, stride=2,padding=1,output_padding=1, bias=False)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2)

        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4,out_channels,3,padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64 # the original channel
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 以下构建残差块， 具体参数可以查看resnet参数表
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 对卷积和与BN层初始化，论文中也提到过
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    # 这里是为了结局两个残差块之间可能维度不匹配无法直接相加的问题，相同类型的残差块只需要改变第一个输入的维数就好，后面的输入维数都等于输出维数
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None

        # 扩维   输入与输出的通道数不匹配，相同才能相加
        if stride != 1 or self.inplanes != block.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, block.expansion*planes,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(block.expansion*planes)
            )

        layers = []
        # 特判第一残差块
        layers.append(block(self.inplanes, planes, stride,downsample=downsample)) # outplane is planes not planes*block.expansion
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x1 = self.relu(x)
        x = self.max_pool(x1)

        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        return x1,x2,x3,x4,x5

class ResUnet(nn.Module):
    def __init__(self,num_classes=1):
        super(ResUnet, self).__init__()
        self.encoder = resnet101(pretrained=True)
        #filters = [64,64, 128, 256, 512] #resnet34 or resnet18
        filters = [64, 256, 512, 1024, 2048] #resnet50,101,152
        self.center = DecoderBlock(in_channels=filters[4],out_channels=filters[3])
        self.decoder4 = DecoderBlock(in_channels=filters[3] + filters[3],out_channels=filters[2])
        self.decoder3 = DecoderBlock(in_channels=filters[2] + filters[2],out_channels=filters[1])
        self.decoder2 = DecoderBlock(in_channels=filters[1] + filters[1],out_channels=filters[0])
        self.decoder1 = DecoderBlock(in_channels=filters[0] + filters[0],out_channels=filters[0])
        self.finalconv = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(32),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1, False),
                                       nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        x1,x2,x3,x4,x5 =self.encoder(x)
        center = self.center(x5)

        d4 = self.decoder4(torch.cat([center, x4], 1))
        d3 = self.decoder3(torch.cat([d4, x3], 1))
        d2 = self.decoder2(torch.cat([d3, x2], 1))
        d1 = self.decoder1(torch.cat([d2, x1], 1))
        d = self.finalconv(d1)
        return d



def resnet18(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet34(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet50(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet101(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model


def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
    return model

if __name__ == '__main__':
    x=torch.rand(1,3,512,512)
    model = ResUnet()
    y = model(x)
    print(y.shape)

