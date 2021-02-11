from torch.utils.data import Dataset
import os
from PIL import Image
from .transforms import RandomCropPaired, RandomLRFlipPaired, RandomUDFlipPaired, ToTensorPaired, NormalizePaired
from torchvision import transforms

import numpy as np
import torch

class HeartDataset(Dataset):
    def __init__(self, dataroot, img_size=256, train=True):
        """
        Dataloader for heart images
        """
        self.img_size = img_size
        self.train    = train
        # get images with labels to train on ct_train_1014_label_322.txt ct_train_1016_image_72.png
        if train:
            # img_names are list of pairs of img name and segmentation masks name
            img_names = [(f.replace('label', 'image') , f) for f in os.listdir(dataroot) if 'label' in f]

            # add root path
            img_names = [(os.path.join(dataroot, i), os.path.join(dataroot, j)) for i, j in img_names]
        else:
            img_names = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith('.png')]
        
        self.img_paths = img_names

        if train:
            trans_fn = [
                RandomLRFlipPaired(),
                RandomUDFlipPaired(),
                RandomCropPaired(self.img_size)
            ]
        else:
            trans_fn  = [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ]
        
        self.transform_fn = transforms.Compose(
            trans_fn
        )
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        if self.train:
            img = Image.open(img_path[0]).resize((self.img_size, self.img_size))
            seg = Image.open(img_path[1]).resize((self.img_size, self.img_size), resample=Image.NEAREST)

            img, seg = self.transform_fn([img, seg])
            
            seg      = torch.from_numpy(np.array(seg))
            img      = transforms.ToTensor()(img)
            img      = transforms.Normalize([0.5], [0.5])(img)
            return [img, seg.to(torch.long)] # no channel for segmentation labels
        else:
            img = Image.open(img_path).resize((self.img_size, self.img_size))
            img = self.transform_fn(img)
            return img

    def __len__(self):
        return len(self.img_paths)
