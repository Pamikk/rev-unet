###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import Mydataset as dataset
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')
def main(args,cfgs):
    #get data config
    for mode in ['train','val','trainval']:
        curset = dataset(cfgs,mode=mode,aug=False)
        loader = DataLoader(curset,batch_size=args.bs,shuffle=True,pin_memory=False)
        for data in tqdm(loader):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs",type=int,default=16,help="batchsize")
    args = parser.parse_args()
    cfgs = cfg()
    main(args,cfgs)
    
    

    