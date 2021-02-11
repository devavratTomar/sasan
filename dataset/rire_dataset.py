from torch.utils.data import Dataset
import os
import skimage.io as io
from .transforms import RandomRotatePaired, NormalizePaired, ToTensorPaired, RandomWidthScalePaired, RandomHeightScalePaired, RandomLRFlipPaired, RandomUDFlipPaired
from torchvision.transforms import Compose
import random

class RireDataset(Dataset):
    def __init__(self, dataroot, file_ext='.png'):
        """
        Other options are just 'CT' or 'MRI'
        """
        self.img_paths = [os.path.join(dataroot, f) for f in os.listdir(dataroot) if f.endswith(file_ext)]
        self.dataset_type = dataset_type
        
        # reshuffle if not paired before concatination so that we get unregistered pairs
        if self.dataset_type  != 'paired':
            random.shuffle(self.img_paths)

        trfs = [
                RandomRotatePaired(),
                RandomWidthScalePaired(),
                RandomHeightScalePaired(),
                RandomLRFlipPaired(),
                RandomUDFlipPaired(),
                ToTensorPaired(),
                NormalizePaired([0.5], [0.5])
            ]
        
        self.transform_fn = Compose(
            trfs
        )
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)
        

        img = [img[:, :256], img[:, 256:512]]          
        img = self.transform_fn(img)
        
        return img

    def __len__(self):
        return len(self.img_paths)

