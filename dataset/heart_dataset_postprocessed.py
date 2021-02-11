import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
from .transforms import RandomRotatePaired, RandomWidthScalePaired, RandomHeightScalePaired, RandomLRFlipPaired, RandomUDFlipPaired, ToTensorPaired

class WholeHeartDatasetPostProcessed(Dataset):
    """
    dataroot assumes the following data structure /images, /labels with same names for corresponding images and labels
    """
    def __init__(self, dataroot, modality, test=False):
        super(WholeHeartDatasetPostProcessed).__init__()
        self.images = []
        self.labels = []
        
        if modality == 'ct':
            self.param1 = -2.8
            self.param2 = 3.2
        elif modality == 'mr':
            self.param1 = -1.8
            self.param2 = 4.4
        else:
            print("Warning no modality selected")
            self.param1 = 0.0
            self.param2 = 1.0

        if type(dataroot) == type(''):
            dataroot = [dataroot]

        for root in dataroot:
            self.images += sorted([os.path.join(root, 'images', f) for f in os.listdir(os.path.join(root, 'images')) if f.endswith('npy')])
            self.labels += sorted([os.path.join(root, 'labels', f) for f in os.listdir(os.path.join(root, 'labels')) if f.endswith('npy')])
        
        if len(self.images) != len(self.labels):
            raise Exception('Images and Labels length do not match')
        
        if not test:
            self.t = transforms.Compose([
                # RandomRotatePaired(),
                RandomLRFlipPaired(),
                RandomUDFlipPaired(),
                ToTensorPaired()])
        else:
            self.t = transforms.Compose([
                ToTensorPaired()])

    def __getitem__(self, index):
        image = np.load(self.images[index])
        label = np.load(self.labels[index])/4.0
        
        # preprocess the data (not fake) if not fake
        if len(label.shape) == 3:
            image = 2.0*(image - self.param1)/(self.param2 - self.param1) - 1.0
            label = label[:, :, 1]
        
        
        image, label  = self.t([image, label])
        
        #image = image*(max_img - min_img) + min_img
        label = label*4
        label = label.round()
        label = label[0]
        label = label.to(torch.long)
        return image.to(torch.float32), label
    
    def __len__(self):
        return len(self.images)