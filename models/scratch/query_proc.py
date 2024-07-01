import sys
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
from torch import nn
from einops import rearrange


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(in_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        # add identity
        x = x + identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, layer_list):
        super(ResNet, self).__init__()

        ResBlock = Bottleneck

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=out_channels)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=out_channels)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=out_channels)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=out_channels)


    def forward(self, x):

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):

        layers = []
        layers.append(ResBlock(planes, planes, i_downsample=None, stride=stride))
        for i in range(blocks - 1):
            layers.append(ResBlock(planes, planes))

        return nn.Sequential(*layers)


def main():

    b, c, h, w = 1, 512, 200, 200
    _input = torch.zeros(size=(b, c, h, w))

    resnet18 = ResNet(in_channels=c, out_channels=int(c/2), layer_list=[2, 2, 2, 2])

    _output = resnet18(_input)

    print(_output.size())


if __name__ == '__main__':
    main()