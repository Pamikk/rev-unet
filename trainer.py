import torch
import torch.nn as nn
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import os
import json
import random
from functools import reduce

from utils import Logger,eval_single_img,save_nib_img
class Trainer:
    def __init__(self,cfg,datasets,net,epoch):
        self.cfg = cfg
        if 'train' in datasets:
            self.trainset = datasets['train']
        if 'val' in datasets:
            self.valset = datasets['val']
        if 'trainval' in datasets:
            self.trainval = datasets['trainval']
        else:
            self.trainval = False
        if 'test' in datasets:
            self.testset = datasets['test']
        self.net = net
        self.calnetParametersNum()

        name = cfg.exp_name
        self.name = name
        self.checkpoints = os.path.join(cfg.checkpoint,name)

        self.device = cfg.device

        self.optimizer = optim.Adam(self.net.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        self.lr_sheudler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,mode='min', factor=cfg.lr_factor, threshold=0.0001,patience=cfg.patience,min_lr=cfg.min_lr)
        
        if not(os.path.exists(self.checkpoints)):
            os.mkdir(self.checkpoints)
        self.predictions = os.path.join(self.checkpoints,'pred')
        if not(os.path.exists(self.predictions)):
            os.mkdir(self.predictions)

        start,total = epoch
        self.start = start        
        self.total = total
        log_dir = os.path.join(self.checkpoints,'logs')
        if not(os.path.exists(log_dir)):
            os.mkdir(log_dir)
        self.logger = Logger(log_dir)
        torch.cuda.empty_cache()
        self.save_every_k_epoch = cfg.save_every_k_epoch #-1 for not save and validate
        self.val_every_k_epoch = cfg.val_every_k_epoch
        self.upadte_grad_every_k_batch = 1

        self.best_Acc = 0
        self.best_Acc_epoch = 0
        self.movingLoss = 0
        self.bestMovingLoss = 10000
        self.bestMovingLossEpoch = 1e9

        self.early_stop_epochs = 50
        self.alpha = 0.95 #for update moving loss
        self.lr_change= cfg.adjust_lr
        self.base_epochs = cfg.base_epochs

        self.save_pred = False
        
        #load from epoch if required
        if start:
            if (start=='-1')or(start==-1):
                self.load_last_epoch()
            else:
                self.load_epoch(start)
        else:
            self.start = 0
        self.net = self.net.to(self.device)

    def load_last_epoch(self):
        files = os.listdir(self.checkpoints)
        idx = 0
        for name in files:
            if name[-3:]=='.pt':
                epoch = name[6:-3]
                if epoch=='best' or epoch=='bestm':
                  continue
                idx = max(idx,int(epoch))
        if idx==0:
            exit()
        else:
            self.load_epoch(str(idx))
    def save_epoch(self,idx,epoch):
        saveDict = {'net':self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler':self.lr_sheudler.state_dict(),
                    'epoch':epoch,
                    'Acc':self.best_Acc,
                    'Acc_epoch':self.best_Acc_epoch,
                    'movingLoss':self.movingLoss,
                    'bestmovingLoss':self.bestMovingLoss,
                    'bestmovingLossEpoch':self.bestMovingLossEpoch}
        path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        torch.save(saveDict,path)                  
    def load_epoch(self,idx):
        model_path = os.path.join(self.checkpoints,'epoch_'+idx+'.pt')
        if os.path.exists(model_path):
            print('load:'+model_path)
            info = torch.load(model_path)
            self.net.load_state_dict(info['net'])
            if not(self.lr_change):
                self.optimizer.load_state_dict(info['optimizer'])#might have bugs about device
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.start = info['epoch']+1
            self.best_Acc = info['Acc']
            self.best_Acc_epoch = info['Acc_epoch']
            self.movingLoss = info['movingLoss']
            self.bestMovingLoss = info['bestmovingLoss']
            self.bestMovingLossEpoch = info['bestmovingLossEpoch']
        else:
            print('no such model at:',model_path)
            exit()
    def _updateRunningLoss(self,loss,epoch):
        if self.bestMovingLoss>loss:
            self.bestMovingLoss = loss
            self.bestMovingLossEpoch = epoch
            print('saving...')
            self.save_epoch('bestm',epoch)
    def calnetParametersNum(self):
        num = 0
        for param in self.net.parameters():
            num += reduce(lambda x,y:x*y,param.shape)
        print(f'number of parameters of network:{num}')
    def logMemoryUsage(self, additionalString=""):
        if torch.cuda.is_available():
            print(additionalString + "Memory {:.0f}Mb max, {:.0f}Mb current".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024, torch.cuda.memory_allocated() / 1024 / 1024))
    def set_lr(self,lr):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']=lr
        #tbi:might set different lr to different kind of parameters
    def adjust_lr(self,lr_factor):
        #adjust learning rate manually
        for param_group in self.optimizer.param_groups:
            param_group['lr']*=lr_factor
    def warm_up(self,epoch):
        if len(self.base_epochs)==0:
            return False
        if epoch <= self.base_epochs[-1]:
            if epoch in self.base_epochs:
                self.adjust_lr(0.1)
            return True
        else:
            return False
    def train_one_epoch(self):
        self.optimizer.zero_grad()
        running_loss ={}
        self.net.train()
        n = len(self.trainset)
        for data in tqdm(self.trainset):
            inputs,labels = data
            labels = labels.to(self.device).float()
            display,loss = self.net(inputs.to(self.device).float(),gts=labels)           
            del inputs,labels
            for k in display.keys():
                if k not in running_loss.keys():
                    running_loss[k] = 0.0
                if np.isnan(display[k]):
                    continue
                else:
                    running_loss[k] += display[k]/n
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            del loss
        self.logMemoryUsage()
        return running_loss
    def train(self):
        print("strat train:",self.name)
        print("start from epoch:",self.start)
        print("=============================")
        self.optimizer.zero_grad()
        print(self.optimizer.param_groups[0]['lr'])
        epoch = self.start
        stop_epochs = 0
        #torch.autograd.set_detect_anomaly(True)
        while epoch < self.total and stop_epochs<self.early_stop_epochs:
            running_loss = self.train_one_epoch()            
            lr = self.optimizer.param_groups[0]['lr']
            self.logger.write_loss(epoch,running_loss,lr)
            #step lr
            
            if not self.warm_up(epoch):
                self.lr_sheudler.step(running_loss['all'])
            lr_ = self.optimizer.param_groups[0]['lr']
            if lr_ == self.cfg.min_lr:
                stop_epochs +=1
            if (epoch+1)%self.save_every_k_epoch==0:
                self.save_epoch(str(epoch),epoch)
            if (epoch+1)%self.val_every_k_epoch==0:                
                metrics = self.validate(epoch,'val',self.save_pred)
                tosave = metrics.keys()
                self.logger.write_metrics(epoch,metrics,tosave)
                Acc = metrics['dice']
                if Acc >= self.best_Acc:
                    self.best_Acc = Acc
                    self.best_Acc_epoch = epoch
                    print("best so far, saving......")
                    self.save_epoch('best',epoch)
                if self.trainval:
                    metrics = self.validate(epoch,'train',self.save_pred)
                    self.logger.write_metrics(epoch,metrics,tosave,mode='Trainval')
            self._updateRunningLoss(running_loss['all'],epoch)
            print(f"best so far with {self.best_Acc} at epoch:{self.best_Acc_epoch}")
            epoch +=1
                
        print("Best Acc: {:.4f} at epoch {}".format(self.best_Acc, self.best_Acc_epoch))
        self.save_epoch(str(epoch-1),epoch-1)
    def validate(self,epoch,mode,save=False):
        self.net.eval()
        print('start Validation Epoch:',epoch)
        if mode=='val':
            valset = self.valset
        else:
            valset = self.trainval
        metrics = {}
        with torch.no_grad():
            for data in tqdm(valset):
                inputs,labels = data
                pds = self.net(inputs.to(self.device).float())
                nB = pds.shape[0]             
                for b in range(nB):
                    metric_ = eval_single_img(pds[b],labels[b])
                    for k in metric_:
                        if k not in (metrics.keys()):
                            metrics[k] = []
                        if not np.isnan(metric_[k]):
                            metrics[k].append(metric_[k])
        for k in metrics:
            metrics[k] = np.mean(metrics[k])
        return metrics
    def test(self):
        self.net.eval()
        res = []
        with torch.no_grad():
            for data in tqdm(self.testset):
                inputs,offsets,names = data
                pds = self.net(inputs.to(self.device).float())
                nB = pds.shape[0]
                for i in range(nB):
                    xOffset, yOffset, zOffset = offsets[i]
                    c,h,w,d = pds[i].shape
                    nh,nw,nd = h+xOffset,w+yOffset,d+zOffset
                    result = torch.zeros([c,nh,nw,nd])
                    result[:,xOffset:xOffset+h,yOffset:yOffset+w,zOffset:zOffset+d]=pds[i]
                    npResult = result.cpu().numpy()
                    save_nib_img(self.predictions,names[i],npResult)
        return res
    def validate_random(self):
        raise NotImplementedError
        self.net.eval()
        self.valset.shuffle = False
        bs = self.valset.batch_size
        imgs = list(range(bs))
        preds = list(range(bs))
        gts = list(range(bs))
        sizes = list(range(bs))
        with torch.no_grad():
            inputs,labels,info = next(iter(self.valset))
            pds = self.net(inputs.to(self.device).float())         
        return imgs,preds,gts,sizes

        


                


        




