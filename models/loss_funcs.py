import torch.nn as nn
import torch
import numpy as np

#Functional Utils
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def dice_loss1d(pd,gt):
    assert pd.shape == gt.shape
    if gt.shape[0]==0:
        return 0
    inter = torch.sum(pd*gt)
    pd_area = torch.sum(torch.pow(pd,2))
    gt_area = torch.sum(torch.pow(gt,2))
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0],device = pd.device)
    return 1-dice.mean()
class LossAPI(nn.Module):
    def __init__(self,cfg):
        super(LossAPI,self).__init__()
        self.loss = Losses[cfg.loss]
    def forward(self,out,gt=None):
        if (gt is None):
            return torch.sigmoid(out)
        else:
            return self.loss(out,gt)
Losses = {}



        







        