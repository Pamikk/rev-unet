import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
import os 
import json
from tqdm import tqdm

def init_weights(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight.data)
        #print(m)
    elif type(m) == torch.nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        #print(m)
def to_cpu(tensor):
    return tensor.detach().cpu()