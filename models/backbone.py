import torch.nn as nn
import math
import os
import numpy as np
import torch
import torch.nn.functional as F

import revtorch.revtorch as rv
DROPOUT = 0.2
LReLU = 0.01
__all__ = ['Coders','BasicBlock']
def conv3x3(in_channels, out_channels, stride=1,bias=False):
    "3x3 convolution with padding"
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=1, bias=bias)

def conv1x1(in_channels,out_channels,stride=1,bias=False):
    return nn.Conv3d(in_channels,out_channels,kernel_size=1,stride=stride,bias=bias)

#bias will be added in normalization layer
class BasicBlock(nn.Module):
    def __init__(self,in_channels,channels):
        super(BasicBlock,self).__init__()
        self.conv = conv3x3(in_channels,channels)
        self.norm = nn.InstanceNorm3d(channels)
        self.relu = nn.LeakyReLU(LReLU)
    def forward(self,x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu(y)
        return y
class BasicBlock_bn(nn.Module):
    def __init__(self,in_channels,channels):
        super(BasicBlock_bn,self).__init__()
        self.conv = conv3x3(in_channels,channels)
        self.norm = nn.BatchNorm3d(channels)
        self.relu = nn.LeakyReLU(LReLU,inplace=True)
    def forward(self,x):
        y = self.conv(x)
        y = self.norm(y)
        y = self.relu(y)
        return y
class Encoder(nn.Module):
    def __init__(self,in_channel,channels,depth=1,dropout=True):
        super(Encoder,self).__init__()
        self.downsample = in_channel == channels
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size = 2)
            self.conv = conv1x1(in_channel,channels)
        if dropout:
            self.dropout = nn.Dropout3d(p=DROPOUT)
        seq=[BasicBlock(channels,channels) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
    def forward(self,x):
        if self.downsample:
            x = self.pool(x)
            x = self.conv(x)
        if not self.dropout is None:
            x = self.dropout(x)
        return self.seq(x)
class Encoder_bn(Encoder):
    def __init__(self,in_channel,channels,depth=1,dropout=True):
        super(Encoder_bn,self).__init__(in_channel,channels,depth,dropout)
        seq=[BasicBlock_bn(channels,channels) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
class Encoder_rev(Encoder):
    def __init__(self,in_channel,channels,depth=1,dropout=True):
        super(Encoder_rev,self).__init__(in_channel,channels,depth,dropout)
        seq = []
        channels = channels//2
        for _ in range(depth):
            f_block = BasicBlock(channels,channels)
            g_block = BasicBlock(channels,channels)
            seq.append(rv.ReversibleBlock(f_block,g_block))
        self.seq = rv.ReversibleSequence(nn.ModuleList(seq))
class Encoder_rev_bn(Encoder):
    def __init__(self,in_channel,channels,depth=1,dropout=True):
        super(Encoder_rev_bn,self).__init__(in_channel,channels,depth,dropout)
        seq = []
        channels = channels//2
        for _ in range(depth):
            f_block = BasicBlock_bn(channels,channels)
            g_block = BasicBlock_bn(channels,channels)
            seq.append(rv.ReversibleBlock(f_block,g_block))
        self.seq = rv.ReversibleSequence(nn.ModuleList(seq))
class Decoder(nn.Module):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder,self).__init__()
        self.upsample = out_channel == channels
        if self.upsample:
            self.conv = conv1x1(channels,out_channel)
            self.up = nn.Upsample(scale_factor=2,mode='trilinear',align_corners=False)
        seq=[BasicBlock(channels,channels) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
    def forward(self,x):
        x = self.seq(x)
        if self.upsample:
            x = self.conv(x)
            x = self.up(x)
        return x
class Decoder_bn(Decoder):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder_bn,self).__init__(out_channel,channels,depth)
        seq=[BasicBlock_bn(channels,channels) for _ in range(depth)]
        self.seq = nn.Sequential(*seq)
        
class Decoder_rev(Decoder):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder_rev,self).__init__(out_channel,channels,depth)
        seq = []
        channels = channels//2
        for _ in range(depth):
            f_block = BasicBlock(channels,channels)
            g_block = BasicBlock(channels,channels)
            seq.append(rv.ReversibleBlock(f_block,g_block))
        self.seq = rv.ReversibleSequence(nn.ModuleList(seq))
class Decoder_rev_bn(Decoder):
    def __init__(self,channels,out_channel,depth=1):
        super(Decoder_rev_bn,self).__init__(out_channel,channels,depth)
        seq = []
        channels = channels//2
        for _ in range(depth):
            f_block = BasicBlock_bn(channels,channels)
            g_block = BasicBlock_bn(channels,channels)
            seq.append(rv.ReversibleBlock(f_block,g_block))
        self.seq = rv.ReversibleSequence(nn.ModuleList(seq))
Coders = {'base':(Encoder,Decoder),'bn':(Encoder_bn,Decoder_bn),'rev':(Encoder_rev,Decoder_rev),'rev_bn':(Encoder_rev_bn,Decoder_rev_bn)}
