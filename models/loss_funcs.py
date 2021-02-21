import torch.nn as nn
import torch
import numpy as np

#Functional Utils
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def dice_loss1d(pd,gt,threshold=0.5):
    assert pd.shape == gt.shape
    if gt.shape[0]==0:
        return 0
    inter = torch.sum(pd*gt)
    pd_area = torch.sum(torch.pow(pd,2))
    gt_area = torch.sum(torch.pow(gt,2))
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0])
    return 1-dice.mean()
class LossAPI(nn.Module):
    def __init__(self,cfg,loss):
        super(LossAPI,self).__init__()
        self.not_match = 0

    def forward(self,outs,gt=None,size=None,infer=False):
        raise NotImplementedError
        if infer:
            return
        else:
            return 
    def reset_notmatch(self):
        self.not_match = 0
Losses = {}



        







        