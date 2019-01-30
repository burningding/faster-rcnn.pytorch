from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.faster_rcnn.resnet import BasicBlock
from model.roi_pooling.modules.roi_pool import _RoIPooling

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
           padding=0, bias=False)

class Generator(nn.Module):
    def __init__(self, inplanes, planes, blocks, stride=1, downsample=None):
        super(Generator, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.conv2 = conv3x3(planes, planes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.k_blocks = self._make_layer(BasicBlock, planes, blocks)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.RCNN_roi_pool(x)
        x = self.k_blocks(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4096, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 2),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            input_flat = input.view(input.size(0), -1)
            output = self.main(input_flat)

        return output.view(-1, 1).squeeze(1)