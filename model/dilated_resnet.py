
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: speedinghzl02
## updated by: RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##https://github.com/PkuRainBow/OCNet.pytorch/blob/master/network/resnet101_baseline.py
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet50
def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, fist_dilation=1, multi_grid=1 , BatchNorm2d = nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert (len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

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

        out = out + residual
        out = self.relu_inplace(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes ,multi_grid=(1,1,1), BatchNorm2d = nn.BatchNorm2d):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = conv3x3(64, 128)
        self.bn3 = BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True) # change
        self.layer1 = self._make_layer(block, 64, layers[0], BatchNorm2d=BatchNorm2d)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, BatchNorm2d=BatchNorm2d)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2, BatchNorm2d=BatchNorm2d)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=multi_grid, BatchNorm2d=BatchNorm2d)
        self.layer5 = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
                BatchNorm2d(512),
                nn.Dropout2d(0.05)
                )
        self.layer6 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1,BatchNorm2d = nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index%len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #针对resnet101在ocr中的应用所做的特化
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x


def get_resnet101_baseline(pretrained = False, num_classes=19,multi_grid=(1,1,1),BatchNorm = nn.BatchNorm2d):
    model = ResNet(Bottleneck,[3, 4, 23, 3], num_classes,multi_grid=multi_grid, BatchNorm2d=BatchNorm)
    if pretrained:
        #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        pretrain_dict = torch.load("initmodel/resnet101_v2.pth")
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model

def get_resnet50_baseline(pretrained = False,num_classes = 19,multi_grid = (1,1,1), BatchNorm = nn.BatchNorm2d):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes,multi_grid=multi_grid, BatchNorm2d=BatchNorm)
    if pretrained:
        #pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        pretrain_dict = torch.load("initmodel/resnet50_v2.pth")
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = get_resnet50_baseline(pretrained=True)
    input = torch.rand(1,3,64,64)
    model.eval()
    model(input)
