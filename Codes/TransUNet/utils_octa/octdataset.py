import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from config_octa import cfg

class OCTDataset(Dataset):
    output_size = cfg.transunet.img_dim  

    def __init__(self, npz_path, transform = False):
        super().__init__()
        self.transform = transform

        # Load data from .npz file
        data = np.load(npz_path)
        self.imgs = data['img']
        self.masks = data['mask']

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Processing the image
        #get as item
        img = self.imgs[idx]
        mask = self.masks[idx]
        
        # create dict
        sample = {'img': img, 'mask': mask}
        
        # if transform, apply
        if self.transform:
            sample = self.transform(sample)
        
        # continue with transformed data    
        img, mask = sample['img'], sample['mask']

        # covert appropriate shape
        img = img.transpose((2, 0, 1)) 
        img = torch.from_numpy(img.astype('float32'))

        mask = mask.transpose((0, 1, 2))
        mask = torch.from_numpy(mask.astype('float32'))
        
        return {'img': img, 'mask': mask}

    def __len__(self):
        return len(self.imgs)