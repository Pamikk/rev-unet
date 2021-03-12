import torch.nn as nn
import torch
import numpy as np

#Functional Utils
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
def cal_dice_loss(pd,gt):
    #nBxnHxnWxnD
    assert pd.shape == gt.shape
    inter = torch.sum(pd*gt,dim=(1,2,3))
    pd_area = torch.sum(pd*pd,dim=(1,2,3))
    gt_area = torch.sum(gt*gt,dim=(1,2,3))
    dice = (2*inter+1)/(pd_area+gt_area+1)
    #fix nans
    dice[dice != dice] = dice.new_tensor([1.0],device = pd.device)
    return 1-dice.mean()
def dice_loss_sigmoid(pds,gts,res={}):
    pds = torch.sigmoid(pds)
    if gts is None:
        preds = (pds>=0.5).float()
        return preds
    total = torch.tensor(0.0,device=pds.device,dtype=pds.dtype)
    res = {}
    for i in range(pds.shape[1]):
        dice = cal_dice_loss(pds[:,i,...],gts[:,i+1,...])
        res [f'dice_c{i}'] = dice.item()
        total += dice
    res['dice'] = total.item()
    return res,total
def dice_loss_softmax(pds,gts,res={}):
    pds = torch.softmax(pds,dim=1)
    if gts is None:
        val = torch.max(val,keepdim=True)[0]
        preds = (pds == val).float()
        return preds
    total = torch.tensor(0.0,device=pds.device,dtype=pds.dtype)
    for i in range(pds.shape[1]):
        dice = cal_dice_loss(pds[:,i,...],gts[:,i,...])
        res [f'dice_c{i}'] = dice.item()
        total += dice
    res['dice'] = total.item()
    return res,total
    
class LossAPI(nn.Module):
    def __init__(self,cfg):
        super(LossAPI,self).__init__()
        self.loss = Losses[cfg.loss]
        self.weights = [cfg.dice_scale,cfg.bce_scale]
    def forward(self,out,gt=None):
        if gt is None:
            return self.loss[0](out)
        res = {}
        total = torch.tensor(0.0,device=out.device,dtype=out.dtype)
        for idx,loss in enumerate(self.loss):
            res,val = loss(out,gt,res)
            total += self.weights[idx]*val
        res['all'] = total.item()
        return res,total
Losses = {'dice_soft':[dice_loss_softmax],'dice_sig':[dice_loss_sigmoid]}



        







        