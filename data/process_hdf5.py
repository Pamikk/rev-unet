import os
import numpy as np
import gc
import h5py
from skimage import transform
import random
from tqdm import tqdm
import nibabel as nib
import argparse

from utils import *

input_path = '../dataset/Mydataset/train'
save_path= '../dataset/Mydataset/processed'
#Refference:
dimension = 3#3d
BUFFER_SIZE = 5
subset_size = 40
dtype = np.float32
inp_exts = ['']
mask_exts = ['_seg']
tsize = (48, 48, 48)
channel = 1

def get_mode(idx):
    if idx%10>=8: #train/val =4/1
        return 'val'
    else:
        return 'train'
def process_test_data(input_path,save_path,channels,tsize):
    assert len(tsize) == dimension
    hdf5_file = h5py.File(save_path, "w")
    file_list = {'test':[]}
    folders = os.listdir(input_path)
    #random.shuffle(a)
    for _,fname in enumerate(folders):
        file_list['test'].append(fname)
    num = len(file_list['test'])
    print(f'Test Set Size:{num}')
    datasets = {}
    for mode in file_list:
        set_size = len(file_list[mode])
        if set_size>0:
            datasets[f'imgs_{mode}'] = hdf5_file.create_dataset(f'imgs_{mode}',(set_size,)+tuple(tsize)+(channels,),dtype=dtype)
            datasets[f'offset_{mode}'] = hdf5_file.create_dataset(f'offset_{mode}',(set_size,3),dtype=dtype)
    img_list = {'test': []}
    offset_list = {'test': []}
    mH,mW,mD = 0,0,0 #max img size
    for mode in file_list:
        print(f'Start processing {mode} images')
        write_buffer = 0
        count = 0
        for fname in tqdm(file_list[mode]):
            path = os.path.join(input_path,fname)# 
            img,_,img_header = load_niis(path,fname,inp_exts)
            assert img.shape[-1] == channels
            #Analyze
            img,offset = crop_alldim_3d(img)#crop zero volume
            w,h,d = img.shape[:3]
            mW,mH,mD = max(mW,w),max(mH,h),max(mD,d)

            
            
            pixel_size = (img_header.structarr['pixdim'][1],
                          img_header.structarr['pixdim'][2],
                          img_header.structarr['pixdim'][3])
            assert pixel_size == (1.0,1.0,1.0)
            

            img = crop_or_pad_slice_to_size(img, tsize, channels)
            img = normalise_image(img)

            img_list[mode].append(img)
            offset_list[mode].append(offset)

            write_buffer += 1

            if write_buffer >= BUFFER_SIZE:

                counter_to = count + write_buffer
                write_to_hdf5_test(datasets, mode, img_list, offset_list, count, counter_to)
                release_tmp_memory([img_list, offset_list], mode)

                # reset stuff for next iteration
                count = counter_to
                write_buffer = 0

        print('Writing remaining data')
        counter_to = count + write_buffer

        if len(file_list[mode]) > 0:
            write_to_hdf5_test(datasets, mode, img_list, offset_list, count,counter_to)
        release_tmp_memory([img_list, offset_list], mode)
    hdf5_file.close()
    print(mW,mH,mD)
def process_train_data(input_path,save_path,channels,tsize):
    assert len(tsize) == dimension
    hdf5_file = h5py.File(save_path, "w")
    file_list = {'train': [], 'val': []}
    folders = os.listdir(input_path)
    #random.shuffle(a)
    for idx,fname in enumerate(folders):
        mode = get_mode(idx)
        file_list[mode].append(fname)
    file_list['trainval'] = file_list['train'][:subset_size]

    train_num = len(file_list['train'])
    val_num = len(file_list['val'])
    print(f'Train Set Size:{train_num}',f'Val Set Size:{val_num}')
    datasets = {}
    for mode in file_list:
        set_size = len(file_list[mode])
        if set_size>0:
            datasets[f'imgs_{mode}'] = hdf5_file.create_dataset(f'imgs_{mode}',(set_size,)+tuple(tsize)+(channels,),dtype=dtype)
            datasets[f'masks_{mode}'] = hdf5_file.create_dataset(f'masks_{mode}',(set_size,)+tuple(tsize),dtype=np.uint8)
    img_list = {'train': [], 'val': [],'trainval':[]}
    mask_list = {'train': [], 'val': [],'trainval':[]}
    mH,mW,mD = 0,0,0 #max img size
    for mode in file_list:
        print(f'Start processing {mode} images')
        write_buffer = 0
        count = 0
        for fname in tqdm(file_list[mode]):
            path = os.path.join(input_path,fname)# 
            img,_,img_header = load_niis(path,fname,inp_exts)
            assert img.shape[-1] == channels
            mask,_,_ = load_niis(path,fname,mask_exts)
            mask = mask.squeeze()
            #Analyze
            img,mask,_ = crop_alldim_3d(img,mask.copy())#crop zero volume
            w,h,d = img.shape[:3]
            mW,mH,mD = max(mW,w),max(mH,h),max(mD,d)

            
            
            pixel_size = (img_header.structarr['pixdim'][1],
                          img_header.structarr['pixdim'][2],
                          img_header.structarr['pixdim'][3])
            assert pixel_size == (1.0,1.0,1.0)
            

            img = crop_or_pad_slice_to_size(img, tsize, channels)
            mask = crop_or_pad_slice_to_size(mask, tsize)

            img = normalise_image(img)

            img_list[mode].append(img)
            mask_list[mode].append(mask)

            write_buffer += 1

            if write_buffer >= BUFFER_SIZE:

                counter_to = count + write_buffer
                write_to_hdf5(datasets, mode, img_list, mask_list, count, counter_to)
                release_tmp_memory([img_list, mask_list], mode)

                # reset stuff for next iteration
                count = counter_to
                write_buffer = 0

        print('Writing remaining data')
        counter_to = count + write_buffer

        if len(file_list[mode]) > 0:
            write_to_hdf5(datasets, mode, img_list, mask_list, count,counter_to)
        release_tmp_memory([img_list, mask_list], mode)
    hdf5_file.close()
    print(mW,mH,mD)


        
def load_and_process_data(input_path,save_path,channels,val,
                          tsize=None,overwrite=False):
    
    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
    if not val:
        data_path = os.path.join(save_path,f'data_train.hdf5')
        if (not os.path.exists(data_path)) or overwrite:
            print("start to process")            
            process_train_data(input_path,data_path,channels,tsize)
        else:
            print('already exists')
    else:
        data_path = os.path.join(save_path,f'data_test.hdf5')
        if (not os.path.exists(data_path)) or overwrite:
            print("start to process test" )            
            process_test_data(input_path,data_path,channels,tsize)
        else:
            print('already exists')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=str, default='../../dataset/Mydataset', help="dataset path")
    parser.add_argument("--train", type=str, default='train', help="train path")
    parser.add_argument("--test", type=str, default='test', help="val path")
    parser.add_argument("--save", type=str, default='processed', help="save path")
    parser.add_argument("--dim", type=int, default=3, help="image dimension")
    parser.add_argument("--channel", type=int, default=1, help="image dimension")
    parser.add_argument("--val",action='store_true',help='trainset or not')
    parser.add_argument("--overwrite",action='store_true',help='overwrite or not')
    parser.add_argument("--tsize", type=tuple, default=None, help="target size")
    args = parser.parse_args()
    dimension = args.dim
    save_path = os.path.join(args.pr,args.save)
    input_path = os.path.join(args.pr,args.train) if not args.val else os.path.join(args.pr,args.test)
    tsize = args.tsize if not args.tsize is None else tsize
    load_and_process_data(input_path,save_path,args.channel,args.val,tsize,args.overwrite)

    

                          
