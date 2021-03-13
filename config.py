
import numpy as np
import random

import json
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.checkpoint='../checkpoints'
        self.cls_num = 2
        self.indices = [0,1,2]        
        self.tsize = (48, 48, 48)
        self.channel = 1
        self.channels = [60,120,240,360][:int(np.log2(min(self.tsize)/3))]+[480] #reduce levels for small resolution
        #3 is the smallest size of feature map
        #highest channel is 480
        self.depth = 2
        
        self.bs = 8       
        self.augment = False
        #train_setting
        self.lr = 0.001
        self.weight_decay=5e-4
        self.momentum = 0.9
        #lr_scheduler
        self.min_lr = 5e-5
        self.lr_factor = 0.25
        self.patience = 12
        #exp_setting
        self.save_every_k_epoch = 15
        self.val_every_k_epoch = 10
        self.adjust_lr = False
        #loss hyp
        self.bce_scale = 1
        self.dice_scale = 1
        self.base_epochs = [-1]#base epochs with large learning rate,adjust lr_facter with 0.1
        if mode=='train':
            self.file_path=f'../dataset/Mydataset/processed/data_train.hdf5'
            self.bs = 32 # batch size
            
            #augmentation parameter
            self.augment = True
            self.aug_num = 3
            self.z_enable = True
            self.flip = True
            self.rot = 10
            self.crop = 0.15
            self.trans = .15
            self.scale = 0.1
            self.intensity = 0.2
            self.elastic = 10
        elif mode=='test':
            self.file = f'./data/test.json'
        
