import torch.utils.data as data
import torch
import json
import h5py
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F

aug_options = ['flip','rot','trans','crop','elastic','intensity']
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
def rotate(src,ang,scale,interp = cv2.INTER_CUBIC):
    if len(src.shape)>3:
        src = src.squeeze()
    h,w = src.shape[:2]
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, scale)
    dst = cv2.warpAffine(src,mat,(w,h),interpolation=interp)
    return dst
def elastic(src,dx,dy,interp=cv2.INTER_CUBIC):
  # elastic deformation is a aug tric also applied in u-net
  # it was described in Best Practices for Convolutional Neural Networks applied to Visual Document Analysis 
  # as random distoration for every pixel
    nx, ny = dx.shape

    grid_y, grid_x = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")

    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    dst = cv2.remap(src, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REFLECT)
    return dst


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
            h,w,d = img.shape[:3]
            if self.cfg.z_enable:
                dims = random.sample(range(3),k=random.randint(1,3))
            else:
                dims = random.sample(range(2),k=random.randint(1,2))
            for idx in dims:
               slices=[slice(0,h),slice(0,w),slice(0,d)]
               for i in range(img.shape[idx]):
                   slices[idx] = slice(i)
                   img[slices[0],slices[1],slices[2],:]= rotate(img[slices[0],slices[1],slices[2],:],ang,scale)
                   mask[slices[0],slices[1],slices[2],:] = rotate(mask[slices[0],slices[1],slices[2],:],ang,scale,interp=cv2.INTER_NEAREST)
        if ('elastic' in augs) and self.elastic:
            h,w,d = img.shape[:3]
            mu = 0
            sigma = random.uniform(0,self.elastic)

            dx = np.random.normal(mu, sigma, (3,3))
            dx_img = cv2.resize(dx, (w,h), interpolation=cv2.INTER_CUBIC)

            dy = np.random.normal(mu, sigma, (3,3))
            dy_img = cv2.resize(dx, (w,h), interpolation=cv2.INTER_CUBIC)

            for z in range(d):
                img[:, :, z, :] = elastic(img[:,:,z,:],dx_img, dy_img,cv2.INTER_CUBIC)
                mask[:, :, z, :] = elastic(mask[:, :, z, :], dx_img, dy_img, cv2.INTER_NEAREST)
            if self.z_enable:
                dx_img = np.zeros((h,w))
                dy = np.random.normal(mu, sigma, (3,3))
                dy_img = cv2.resize(dx, (w,h), interpolation=cv2.INTER_CUBIC)
                for y in range(h):
                    img[y, ...] = elastic(img[y,...],dx_img, dy_img)
                    mask[y, ...] = elastic(mask[y,...], dx_img, dy_img, cv2.INTER_NEAREST)
        if ('intensity' in augs) and self.cfg.intensity:
            for i in range(img.shape[-1]): 
                img[:, :, :, i] *= (1 + np.random.uniform(-self.cfg.intensity,self.cfg.intensity))
        return img,mask

    def __len__(self):
        return len(self.imgs)

    def img_to_tensor(self,img):
        data = torch.tensor(np.transpose(img,[3,0,1,2]),dtype=torch.float)
        return data
    def gen_gts(self,mask):
        #transfer to on-shot
        return mask
        

    def __getitem__(self,idx):
        #print(name)
        if self.mode=='train':
            aug = []
            if self.aug:
                img,mask = self.random_aug_3d(img,mask)
            labels = self.gen_gts(mask)
            data = self.img_to_tensor(img)
            return data,labels      
        else:
            #validation set
            data = self.img_to_tensor(img)
            if self.mode=='val':
                labels = self.gen_gts(mask)
                return data,labels
            else:
                #test
                return data,offsets

                





