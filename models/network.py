import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .backbone import ResNet,conv1x1,conv3x3,Darknet
from .loss_funcs import  LossAPI
from .models import Darknetv3
from .utils import init_weights
networks = {}
def NetAPI(cfg,net,loss,init=True):
    raise NotImplementedError
    if init:
        network.initialization()
    return network

class NonResidual(nn.Module):
    multiple=2
    def __init__(self,in_channels,channels,stride=1):
        super(NonResidual,self).__init__()
        self.conv1 = conv1x1(in_channels,channels,stride)
        self.relu = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels,momentum=0.9, eps=1e-5)
        self.conv2 = conv3x3(channels,channels*NonResidual.multiple)
        self.bn2 = nn.BatchNorm2d(channels*NonResidual.multiple,momentum=0.9, eps=1e-5)
    def forward(self,x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        return y
class No_new_Net(nn.Module):
    def __init__(self,cfg):
        super(No_new_Net,self).__init__()
        self.channels = cfg.channels
        self.out_channel = cfg.cls_num

    




    