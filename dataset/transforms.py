import random
from torchvision import transforms
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from skimage.transform import rotate, AffineTransform, warp, rescale, resize

def crop_above_one(mat):
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i][j] > 1:
                mat[i][j] = 1

class RandomRotatePaired(object):
    def __init__(self, angle_range=(0, 360)):
        self.min_angle = angle_range[0]
        self.max_angle = angle_range[1]
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        rand_angle = random.randrange(self.min_angle, self.max_angle)
        return [rotate(data[0], angle=rand_angle, mode='edge'), rotate(data[1], angle=rand_angle, order=0)]

class RandomWidthScalePaired(object):
    def __init__(self, scale_range=(0.8, 1)):       
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        scale_w = self.min_scale + random.random() * (self.max_scale - self.min_scale)

        # get the size of a new width and make sure it is divisible by 2
        n_w = int(round(data[0].shape[1] * scale_w))
        if n_w % 2 == 1:
            n_w -= 1
        
        # expected difference after scaling on both sides
        diff_w = int((data[0].shape[1] - n_w) / 2)    
        
        if diff_w == 0:
            return [data[0], data[1]]
               
        # create an array to fill the diff with black pixels
        fill_w_0 = []
        fill_value_0 = data[0].min()
        for cnt in range(data[0].shape[0]):
            fill_w_0.append([fill_value_0] * diff_w)
        
        fill_w_1 = []
        fill_value_1 = data[1].min()
        for cnt in range(data[1].shape[0]):
            fill_w_1.append([fill_value_1] * diff_w)

        # resize both images
        resized_mr = resize(data[0],  output_shape=(data[0].shape[0], n_w))    
        resized_ct = resize(data[1], order=0,  output_shape=(data[0].shape[0], n_w))  
        
        # stack them back to their original size
        stacked_mr = np.hstack([fill_w_0, resized_mr, fill_w_0]) 
        stacked_ct = np.hstack([fill_w_1, resized_ct, fill_w_1]) 
        
        # crop above one
        crop_above_one(stacked_mr)
        crop_above_one(stacked_ct)
        return [stacked_mr, stacked_ct]
    
    
class RandomHeightScalePaired(object):
    def __init__(self, scale_range=(0.8, 1)):     
        self.min_scale = scale_range[0]
        self.max_scale = scale_range[1]
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        scale_h = self.min_scale + random.random() * (self.max_scale - self.min_scale)
        
        # get size of new height and make sure it is divisible by two
        n_h = int(round(data[0].shape[0] * scale_h)) 
        if n_h % 2 == 1:
            n_h -= 1
        
        # expected difference after scaling on both sides
        diff_h = int((data[0].shape[0] - n_h) / 2)
        
        
        if diff_h == 0:
            return [data[0], data[1]]
               
        # create an array to fill the diff with black pixels
        fill_h_0 = []
        fill_value_0 = data[0].min()
        for cnt in range(diff_h):
            fill_h_0.append([fill_value_0] * data[0].shape[1])

        fill_h_1 = []
        fill_value_1 = data[1].min()
        for cnt in range(diff_h):
            fill_h_1.append([fill_value_1] * data[1].shape[1])
        
        # resize both images
        resized_mr = resize(data[0],  output_shape=(n_h, data[0].shape[1]))    
        resized_ct = resize(data[1], order=0,  output_shape=(n_h, data[0].shape[1]))  
        
        # stack them back to their original size
        stacked_mr = np.vstack([fill_h_0, resized_mr, fill_h_0]) 
        stacked_ct = np.vstack([fill_h_1, resized_ct, fill_h_1]) 
        
        # crop above one
        crop_above_one(stacked_mr)
        crop_above_one(stacked_ct)
        return [stacked_mr, stacked_ct]  

class RandomLRFlip(object):
    def __init__(self, flip_proba=0.5):
        self.flip_proba = flip_proba
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        do_flip = random.random() < self.flip_proba
    
        if do_flip:
            return np.fliplr(data).copy()
        else:
            return data

class RandomLRFlipPaired(object):
    def __init__(self, flip_proba=0.5):
        self.flip_proba = flip_proba
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        do_flip = random.random() < self.flip_proba
    
        if do_flip:
            return [np.fliplr(data[0]).copy(), np.fliplr(data[1]).copy()]
        else:
            return data

class RandomUDFlipPaired(object):
    def __init__(self, flip_proba=0.5):
        self.flip_proba = flip_proba
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        do_flip = random.random() < self.flip_proba
    
        if do_flip:
            return [np.flipud(data[0]).copy(), np.flipud(data[1]).copy()]
        else:
            return data

class RandomCropPaired(object):
    def __init__(self, img_size):
        self.img_size = img_size
    
    def __call__(self, imgs):
        w, h = self.img_size + 30, self.img_size + 30
        img0 = imgs[0].resize((w, h))
        img1 = imgs[1].resize((w, h), resample=Image.NEAREST)
        
        th, tw = self.img_size, self.img_size

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        out_img0 = F.crop(img0, i, j, self.img_size, self.img_size)
        out_img1 = F.crop(img1, i, j, self.img_size, self.img_size)

        return [out_img0, out_img1]
        

class RandomUDFlip(object):
    def __init__(self, flip_proba=0.5):
        self.flip_proba = flip_proba
        
    def __call__(self, data):
        """We expect data as a dict of images.
        """
        do_flip = random.random() < self.flip_proba
        
        if do_flip:
            return data.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return data        
        

class Normalize(object):
    def __init__(self, m, std):
        self.norm = transforms.Normalize(m, std)
        
    def __call__(self, data):
        return self.norm(data)

class NormalizePaired(object):
    def __init__(self, m, std):
        self.norm = transforms.Normalize(m, std)
        
    def __call__(self, data):
        return [self.norm(data[0]), self.norm(data[1])]
    
class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()
        
    def __call__(self, data):
        
        return self.totensor(data)

class ToTensorPaired(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()
        
    def __call__(self, data):
        
        return [self.totensor(data[0]), self.totensor(data[1])]
