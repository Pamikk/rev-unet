import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .backbone import *
from .loss_funcs import  LossAPI
networks = {}
def NetAPI(cfg,init=True):
    raise NotImplementedError
class NoNewNet(nn.Module):
    def __init__(self,cfg):
        super(NoNewNet,self).__init__()
        channels = cfg.channels
        out_channels = []
        self.out_channel = cfg.cls_num
        encoder,decoder = Coders[cfg.net]
        self.loss = LossAPI(cfg)
        self.in_channel = cfg.channels
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        self.in_conv = BasicBlock(cfg.channel,self.in_channel)
        self.out_conv = BasicBlock(cfg.cls_num,self.in_channel)
        for channel in channels:
            out_channels.insert(0,self.in_channel)
            self.encoders.append(encoder(self.in_channel,channel,depth=cfg.depth))
            self.in_channel = channel
            
        for channel in out_channels:
            self.decoders.append(decoder(self.in_channel,channel))
            self.in_channel = channel
    def forward(self,x,gts=None):
        feats = []
        for i,encoder in enumerate(self.encoders):
            if i!=0:
                feats.insert(0,x)
            x = encoder(x)
            
        for i,decoder in enumerate(self.decoders):
            x = decoder(x)
            if i<(len(feats)):
                x+=feats[i]
        return self.loss(x,gts)



    




    