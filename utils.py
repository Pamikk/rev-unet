import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
import json
import cv2
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure

def tensor_to_img(src):
    dst = np.transpose(src.cpu().numpy(),[1,2,0])
    return dst
class Logger(object):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        print('Epoch',':',epoch,'-',lr)
        writer = SummaryWriter(log_dir=self.log_dir)
        writer.add_scalar('lr',math.log(lr),epoch)
        for k in losses:
            if losses[k]>0:            
                writer.add_scalar('Train/'+k,losses[k],epoch)            
                print(k,':',losses[k])
                #self.writer.flush()
        tmp+= str(round(losses['all'],5))+'\t'
        self.write_line2file('train',tmp)
        writer.close()
    def write_metrics(self,epoch,metrics,save=[],mode='Val',log=True):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        writer = SummaryWriter(log_dir=self.log_dir)
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'
            if log:
                tag = mode+'/'+k            
                writer.add_scalar(tag,metrics[k],epoch)
                #self.writer.flush()
            print(k,':',metrics[k])
        
        self.write_line2file('val',tmp)
        writer.close()
def eval_single_hd95(result, reference):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    modify from medpy _surface_distances
    """
    if 0 == numpy.count_nonzero(result): 
        return np.nan()
    if 0 == numpy.count_nonzero(reference): 
        return np.nan()
    result = numpy.atleast_1d(result.astype(numpy.bool))
    reference = numpy.atleast_1d(reference.astype(numpy.bool))
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, 1)
      
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=None)
    dt_= distance_transform_edt(~result_border, sampling=None)
    hd95 = np.percentile(np.hstack((dt[result_border], dt_[reference_border])), 95)
    return hd95
def eval_single_dice(pd,gt):
    #dice score, sensitivity,specifity
    assert pd.shape==gt.shape
    if (pd.sum()==0) or (gt.sum()==0):
        return 0,0,0
    inters = (pd*gt).sum()
    cover = pd.sum() + gt.sum()
    
    dc = (2*inters+1)/(cover+1)
    precision = (inters+1)/(pd.sum()+1)
    recall = (inters+1)/(gt.sum()+1)
    return dc,precision,recall

def eval_single_img(pds,gts,each_cls=True):
    assert pds.shape==gts.shape
    nC = pds.shape[0]
    metrics = {'dice':0.0,'prec':0.0,'recall':0.0}
    val = 10e9
    for i in range(nC):
        dc,p,r = eval_single_dice(pds[i,...],gts[i,...])
        hd95 = eval_single_dice(pds[i,...],gts[i,...])
        if (each_cls):
            metrics[f'dice_c{i}'] = dc
            metrics[f'prec_c{i}'] = p
            metrics[f'recall_c{i}'] = r
            metrics[f'hd95_c{i}'] = hd95
        if i>0:
            metrics['dice'] += dc/nC
            metrics['prec'] += p/nC
            metrics['recall'] += r/nC
            if not(np.isnan(hd95)):
                val = min(val,hd95)
    metrics['hd95'] = val
    return metrics
                

        















