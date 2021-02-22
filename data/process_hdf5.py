import os
import numpy as np
import gc
import h5py
from skimage import transform
import random
from tqdm import tqdm
import nibabel as nib
import argparse

input_path = '../dataset/Mydataset/train'
save_path= '../dataset/Mydataset/processed'
#Refference:
dimension = 3#3d
BUFFER_SIZE = 5
subset_size = 200
dtype = np.float32
inp_exts = ['']
mask_exts = ['_seg']
def crop_alldim_3d(image, mask=None):
    h,w,d,_ = image.shape

    coords = np.argwhere(image > 0)
    x0, y0, z0, _ = coords.min(axis=0)
    x1, y1, z1, _ = coords.max(axis=0) + 1
    dx = min(x0,h-x1)
    dy = min(y0,w-y1)
    dz = min(z0,d-z1)
    image = image[dx:h-dx, dy:w-dy, dz:d-dz, :]
    #clip symmetrically to keep the center
    if not mask is None:
        return image, mask[x0:x1, y0:y1, z0:z1]
    return image
def load_niis(path,name,exts=['']):
    imgs = []
    affines = []
    headers = []
    for ext in exts:
        im_path = os.path.join(path,name+ext+'.nii.gz')
        if not os.paht.exists(im_path):
            im_path = os.path.join(path,name,name+ext+'.nii.gz')
            #sometimes image are packed in sub dir
            if not os.paht.exists(im_path):
                # no image matched, continue
                continue
        nimg = nib.load(im_path)
        imgs.append(nimg.get_data())
        affines.append(nimg.affine)
        headers.append(nimg.header)
    return np.stack(imgs,-1),affines[0],headers[0] 
def get_mode
def process_train_data(input_path,save_path,channels,tsize,target_resolution):
    assert len(tsize) == dimension
    assert len(target_resolution) == dimension
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
            datasets[f'imgs_{mode}'] = hdf5_file.create_dataset(f'imgs_{mode}',(set_size)+tuple(tsize)+(channels,),dtype=dtype)
            datasets[f'masks_{mode}'] = hdf5_file.create_dataset(f'imgs_{mode}',(set_size)+tuple(tsize),dtype=unint8)
    img_list = {'train': [], 'val': [],'trainval':[]}
    mask_list = {'train': [], 'val': [],'trainval':[]}
    mH,mW,mD = 0,0,0 #max img size
    mHc,mWc,mWc = 0,0,0 #max cropped img size
    count = 0
    for mode in file_list:
        print('Start processing {mode} images')
        for fname in tqdm(file_list[mode]):
            path = os.path.join(input_path,fname)# 
            img,_,img_header = load_niis(path,fname,inp_exts)
            assert img.shape[-1] == channels
            mask,_,_ = load_niis(path,fname,mask_exts)
            mask = mask.squeeze()
            #Analyze
            w,h,d = img.shape[:3]
            mW,mH,mD = (mW,w),(mH,h),(mD,d)

            img,mask = crop_alldim_3d(img,mask.copy())#crop zero volume

            w,h,d = img.shape[:3]
            mWc,mHc,mDc = (mWc,w),(mHc,h),(mDc,d)

            psize = (img_header.structarr['pixdim'][1],
                       img_header.structarr['pixdim'][2],
                       img_header.structarr['pixdim'][3])
            scale_vector = [psize[0] / target_resolution[0],
                            psize[1] / target_resolution[1],
                            psize[2]/ target_resolution[2]]

        
def load_and_process_data(input_path,save_path,channels,mode='train',
                          tsize=None,target_resolution=None,overwrite=False):
    target_name = f'data_{mode}.hdf5'
    
    data_path = os.path.join(save_path,target_name)

    if not(os.path.exists(save_path)):
        os.mkdir(save_path)
    if not os.path.exists(data_path) or overwrite:
        if mode=='train':
            process_train_data(input_path,data_path,channels,tsize,target_resolution)
        elif mode=='val':
            pass
            #process_val_data(input_path,data_path,channels,tsize)
    else:
        print('already exists')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=str, default='../dataset/Mydataset', help="dataset path")
    parser.add_argument("--train", type=str, default='train', help="train path")
    parser.add_argument("--val", type=str, default='val', help="val path")
    parser.add_argument("--save", type=str, default='processed', help="save path")
    parser.add_argument("--dim", type=int, default=3, help="image dimension")
    parser.add_argument("--channel", type=int, default=1, help="image dimension")
    parser.add_argument("--test",action='store_true',help='generate test set ')
    parser.add_argument("--overwrite",action='store_true',help='overwrite or not')
    args = parser.parse_args()
    dimension = args.dim
    save_path = os.path.join(args.pr,args.save)
    input_train_path = os.path.join(args.pr,args.train)
    input_val_path = os.path.join(args.pr,args.val)

                          
