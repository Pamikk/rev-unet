import torch.utils.data as data
import torch
import json
import h5py
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F

aug_options = ['flip','rot','trans','crop']
#stack functions for collate_fn
#Notice: all dicts need have same keys and all lists should have same length
def stack_dicts(dicts):
    if len(dicts)==0:
        return None
    res = {}
    for k in dicts[0].keys():
        res[k] = [obj[k] for obj in dicts]
    return res

def stack_list(lists):
    if len(lists)==0:
        return None
    res = list(range(len(lists[0])))
    for k in range(len(lists[0])):
        res[k] = torch.stack([obj[k] for obj in lists])
    return res
def rand(item):
    try:
        tmp=[]
        for i in item:
            tmp.append(random.uniform(-i,i))
    except:
        if random.random()<0.5:
            return random.uniform(-i,i)
        else:
            return 0
    finally:
        return tuple(tmp)   
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_LINEAR)
    return dst
def translate_3d(src,mask,trans,z_enable=True):
    h,w,d = src.shape[:3]
    tx = int(random.uniform(-w*trans,w*trans))
    ty = int(random.uniform(-h*trans,h*trans))
    if z_enable:
        tz = int(random.uniform(-d*trans,d*trans))
    else:
        tz = 0
    dsx = slice(max(tx,0),min(tx+w,w))
    dsy = slice(max(ty,0),min(ty+h,h))
    dsz = slice(max(tz,0),min(tz+d,d))
    srx = slice(max(-tx,0),min(-tx+w,w))
    sry = slice(max(-ty,0),min(-ty+h,h))
    srz = slice(max(-tz,0),min(-tz+d,d))
    dst = np.zeros_like(src)
    mask_dst = np.zeros_like(mask)
    dst[dsy,dsx,dsz,:] = src[sry,srx,srz,:]
    mask_dst[dsy,dsx,dsz,:] = mask[sry,srx,srz,:]
    return dst,mask_dst
    
    return dst,labels
def crop_3d(src,mask,crop,z_enable=True):
    h,w,d = src.shape[:3]
    txm = int(random.uniform(0,w*crop))
    tym = int(random.uniform(0,h*crop))
    if z_enable:
        txm = int(random.uniform(0,d*crop))
        tzmx = int(random.uniform(d*(1-crop),d-0.1))
    else:
        txm = 0
        tzmx = d-1    
    txmx = int(random.uniform(w*(1-crop),w-0.1))
    tymx = int(random.uniform(h*(1-crop),h-0.1))
    dst = (src[tym:tymx+1,txm:txmx+1,tzm:tzmx+1,:]).copy()
    mask_dst = (mask[tym:tymx+1,txm:txmx+1,tzm:tzmx+1,:]).copy()
    return dst,mask
def rotate(src,mask,ang,scale):
    if len(src.shape)>3:
        src = src.squeeze()
    h,w = src.shape[:2]
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, scale)
    dst = cv2.warpAffine(src,mat,(w,h),interpolation=cv2.INTER_CUBIC)
    mask_dst = cv2.warpAffine(mask,mat,(w,h),interpolation=cv2.INTER_NEAREST)
    return dst,mask_dst
def flip(src,mask,dim):
    dst = np.flip(src,dim)
    mask =np.flip(mask,dim)
    return dst,mask
class Mydataset(data.Dataset):
    def __init__(self,cfg,mode='train'):
        self.cfg = cfg
        data = h5py.File(cfg.file_path)
        self.mode = mode
        self.accm_batch = 0
        self.size = cfg.size
    def random_aug_3d(self,img,mask):
        assert len(img.shape) == 4
        assert len(mask.shape) == 4
        augs = random.sample(aug_options,k=random.randint(0,self.aug_num))
        if ('flip' in augs) and self.cfg.flip:
            if self.cfg.z_enable:
                dims = random.sample(range(3),k=random.randint(1,3))
            else:
                dims = random.sample(range(3),k=random.randint(1,2))
            for d in dims:  
                img,mask = flip(img,mask,dim=d)
        if ('trans' in augs) and self.cfg.trans:
            img,mask = translate(img,mask,self.cfg.trans,self.cfg.z_enable)
        if ('crop' in augs) and self.cfg.crop:
            img,mask = crop(img,mask,self.cfg.crop,self.cfg.z_enable)
        if ('rot' in augs) and self.cfg.rot:
            ang = random.uniform(-self.cfg.rot,self.cfg.rot)
            scale = random.uniform(1-self.cfg.scale,1+self.cfg.scale)
            if self.cfg.z_enable:
                dims = random.sample(range(3),k=random.randint(1,3))
            else:
                dims = random.sample(range(3),k=random.randint(1,2))
            for d in dims:
               slices
               img,mask= rotate(img,mask,ang,scale)
    def __len__(self):
        return len(self.imgs)

    def img_to_tensor(self,img):
        data = torch.tensor(np.transpose(img,[2,0,1]),dtype=torch.float)
        if data.max()>1:
            data /= 255.0
        return data
    def gen_gts(self,anno):
        gts = torch.zeros((anno['obj_num'],ls+4),dtype=torch.float)
        if anno['obj_num'] == 0:
            return gts
        labels = torch.tensor(anno['labels'])[:,:ls+4]
        assert labels.shape[-1] == ls+4
        gts[:,0] = labels[:,0]
        gts[:,ls] = (labels[:,ls]+labels[:,ls+2])/2
        gts[:,ls+1] = (labels[:,ls+1]+labels[:,ls+3])/2
        gts[:,ls+2] = (labels[:,ls+2]-labels[:,ls])
        gts[:,ls+3] = (labels[:,ls+3]-labels[:,ls+1])
        return gts
        
    def normalize_gts(self,labels,size):
        #transfer
        if len(labels)== 0:
            return labels
        labels[:,ls:]/=size 
        return labels

    def pad_to_square(self,img):
        h,w,_= img.shape
        ts = max(h,w)
        diff1 = abs(h-ts)
        diff2 = abs(w-ts)
        pad = (diff1//2,diff2//2,diff1-diff1//2,diff2-diff2//2)
        img = cv2.copyMakeBorder(img,pad[0],pad[2],pad[1],pad[3],cv2.BORDER_CONSTANT,0)
        return img,(pad[0],pad[1])

    def __getitem__(self,idx):
        name = self.imgs[idx]
        anno = self.annos[name]
        img = cv2.imread(os.path.join(self.img_path,name+'.jpg'))
        ##print(img.shape)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        h,w = img.shape[:2]        
        labels = self.gen_gts(anno)
        #print(name)
        if self.mode=='train':
            aug = []
            if self.aug:
                
            img,pad = self.pad_to_square(img)
            size = img.shape[0]
            labels[:,ls]+= pad[1]
            labels[:,ls+1]+= pad[0]
            data = self.img_to_tensor(img)
            labels = self.normalize_gts(labels,size)
            return data,labels      
        else:
            #validation set
            img,pad = self.pad_to_square(img)
            img = resize(img,(self.cfg.size,self.cfg.size))
            data = self.img_to_tensor(img)
            info ={'size':(h,w),'img_id':name,'pad':pad}
            if self.mode=='val':
                return data,labels,info
            else:
                return data,info
    def collate_fn(self,batch):
        if self.mode=='test':
            data,info = list(zip(*batch))
            data = torch.stack(data)
            info = stack_dicts(info)
            return data,info 
        elif self.mode=='val':
            data,labels,info = list(zip(*batch))
            info = stack_dicts(info)
            data = torch.stack(data)
        elif self.mode=='train':
            data,labels = list(zip(*batch))
            if (self.accm_batch % 10 == 0)and (self.aug):
                self.size = random.choice(self.cfg.sizes)
            tsize = (self.size,self.size)
            self.accm_batch += 1
            data = torch.stack([F.interpolate(img.unsqueeze(0),tsize,mode='bilinear').squeeze(0) for img in data]) #multi-scale-training   
        tmp =[]
                   
                
        for i,bboxes in enumerate(labels):
            if len(bboxes)>0:
                label = torch.zeros(len(bboxes),ls+5)
                
                label[:,0] = i
                tmp.append(label)
        if len(tmp)>0:
            labels = torch.cat(tmp,dim=0)
            labels = labels.reshape(-1,ls+5)
        else:
            labels = torch.tensor(tmp,dtype=torch.float).reshape(-1,ls+5)
        if self.mode=='train':
            return data,labels
        else:
            return data,labels,info

                





