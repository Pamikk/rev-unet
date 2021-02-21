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

class Network(nn.Module):
    def __init__(self,cfg,loss):
        super(Network,self).__init__()
        self.path = os.path.join(cfg.pre_trained_path,'yolov3.weights')
        self.encoders = ''
        decoders = []
        self.decoders = nn.ModuleList(decoders)
        self.loss = LossAPI(cfg,loss)
    def initialization(self):
        raise NotImplementedError
    def make_prediction(self,out_channel,block,channel,upsample=True):
        if upsample:
            upsample = nn.Sequential(conv1x1(self.in_channel,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),
                                           self.relu,Upsample(scale_factor=2,mode='nearest'))
            cat_channel = self.out_channels.pop(0)
            self.in_channel = channel + cat_channel
        else:
            upsample = nn.Identity()
        decoders=[block(self.in_channel,channel),block(channel*block.multiple,channel)]
        decoders.append(nn.Sequential(conv1x1(channel*block.multiple,channel),nn.BatchNorm2d(channel,momentum=0.9, eps=1e-5),self.relu))        
        pred = nn.Sequential(conv3x3(channel,channel*block.multiple),nn.BatchNorm2d(channel*block.multiple,momentum=0.9, eps=1e-5),self.relu,
                conv1x1(channel*block.multiple,out_channel,bias=True))
        self.in_channel = channel
        return nn.ModuleList([upsample,nn.Sequential(*decoders),pred])
    def forward(self,x,optimizer=None,gts=None):
        size = x.shape[-2:]
        feats = self.encoders(x)
        outs = list(range(len(self.decoders)))
        x = feats.pop(0)
        y = []
        for i,decoders in enumerate(self.decoders):
            up,decoder,pred = decoders
            x = torch.cat([up(x)]+y,dim=1)
            x = decoder(x)
            out = pred(x)
            outs[i] = out
            y = [feats.pop(0)]
        if self.training:
            display,loss = self.loss(outs,gts,size)
            if optimizer!=None:
                # for network like GAN
                pass
            else:          
                return display,loss
        else:
            return  self.loss(outs,size=size,infer=True)

    




    