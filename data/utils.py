import os
import numpy as np
import gc
import h5py
from skimage import transform
import random
from tqdm import tqdm
import nibabel as nib
def crop_alldim_3d(image, mask=None):
    w,h,d,_ = image.shape

    coords = np.argwhere(image > 0)
    x0, y0, z0, _ = coords.min(axis=0)
    x1, y1, z1, _ = coords.max(axis=0) + 1
    dx = min(x0,w-x1)
    dy = min(y0,h-y1)
    dz = min(z0,d-z1)
    image = image[dx:h-dx, dy:w-dy, dz:d-dz, :]
    #clip symmetrically to keep the center
    if not mask is None:
        return image, mask[x0:x1, y0:y1, z0:z1],(dx,dy,dz)
    return image,(dx,dy,dz)
def crop_or_pad_slice_to_size(image, target_size, channels=None):
    '''
    Make sure that the image has the desired dimensions
    '''

    
    x_t, y_t, z_t = target_size[0:3]
    x_s, y_s, z_s = image.shape[0:3]

    if not channels is None:
        output_volume = np.zeros((x_t, y_t, z_t, channels), dtype=np.float32)
    else:
        output_volume = np.zeros((x_t, y_t, z_t), dtype=np.float32)

    x_d = abs(x_t - x_s) // 2
    y_d = abs(y_t - y_s) // 2
    z_d = abs(z_t - z_s) // 2

    t_ranges = []
    s_ranges = []

    for t, s, d in zip([x_t, y_t, z_t], [x_s, y_s, z_s], [x_d, y_d, z_d]):

        if t < s:
            t_range = slice(t)
            s_range = slice(d, d + t)
        else:
            t_range = slice(d, d + s)
            s_range = slice(s)

        t_ranges.append(t_range)
        s_ranges.append(s_range)

    if not channels is None:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2], :] = image[s_ranges[0], s_ranges[1], s_ranges[2], :]
    else:
        output_volume[t_ranges[0], t_ranges[1], t_ranges[2]] = image[s_ranges[0], s_ranges[1], s_ranges[2]]

    return output_volume
def normalise_image(image):
    '''
    standardize based on nonzero pixels
    '''
    m = np.nanmean(np.where(image == 0, np.nan, image), axis=(0, 1, 2)).astype(np.float32)
    s = np.nanstd(np.where(image == 0, np.nan, image), axis=(0,1,2)).astype(np.float32)
    normalized = np.divide((image - m), s)
    image = np.where(image == 0, 0, normalized)
    return image
def write_to_hdf5(dataset, mode, img_list, mask_list, start,end):

    img_arr = np.asarray(img_list[mode], dtype=np.float32)
    mask_arr = np.asarray(mask_list[mode], dtype=np.uint8)
    
    dataset[f'imgs_{mode}'][start:end, ...] = img_arr
    dataset[f'masks_{mode}'][start:end, ...] = mask_arr
def write_to_hdf5_test(dataset, mode, img_list, offset_list, start,end):

    img_arr = np.asarray(img_list[mode], dtype=np.float32)
    offset_arr = np.asarray(offset_list[mode], dtype=np.float32)
    dataset[f'imgs_{mode}'][start:end, ...] = img_arr
    dataset[f'offset_{mode}'][start:end, ...] = offset_arr
def release_tmp_memory(lists,mode):
    '''
    Helper function to reset the tmp lists and free the memory
    '''
    for tmp in lists:
        tmp[mode].clear()
    gc.collect()
def load_niis(path,name,exts=['']):
    imgs = []
    affines = []
    headers = []
    for ext in exts:
        im_path = os.path.join(path,name+ext+'.nii.gz')
        if not os.path.exists(im_path):
            im_path = os.path.join(path,name,name+ext+'.nii.gz')
            #sometimes image are packed in sub dir
            if not os.path.exists(im_path):
                # no image matched, continue
                continue
        nimg = nib.load(im_path)
        imgs.append(nimg.get_fdata())
        affines.append(nimg.affine)
        headers.append(nimg.header)
    return np.stack(imgs,-1),affines[0],headers[0] 