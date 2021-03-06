
import numpy as np
import random
import json
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.checkpoint='../checkpoints'
        self.cls_num = 20        
        self.tsize = (48, 64, 48)
        self.channel = 1
        
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
        self.obj_scale = 2
        self.noobj_scale = 5
        self.cls_scale = 1
        self.reg_scale = 1#for giou
        self.ignore_threshold = 0.5
        self.match_threshold = 0#regard as match above this threshold
        self.base_epochs = [-1]#base epochs with large learning rate,adjust lr_facter with 0.1
        if mode=='train':
            self.file_path=f'./data/train.json'
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
            
        elif mode=='val':
            self.file = f'./data/val.json'
        elif mode=='trainval':
            self.file = f'./data/trainval.json'
        elif mode=='test':
            self.file = f'./data/test.json'
        
