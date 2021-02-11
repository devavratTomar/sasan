from utilities.util import load_network

import torch
import torch.nn as nn

from PIL import Image
import os

import numpy as np
import medpy.io as medio
from medpy import metric

# sasan models
from models import UnetAttention, GeneratorED, SegmentationModel
# from nnunet.network_architecture.generic_UNet import Generic_UNet
# from nnunet.network_architecture.initialization import InitWeights_He

# cycle gan, ugatit models
from models_baseline import CycleGANGenerator, UGATITResnetGenerator

from .utils import overlay_seg_img

INPUT_CH = 3
N_ATTENTIONS = 8
N_CLASSES = 5

class SegAccuracyUpperBound(object):
    def __init__(self, model_name, test_folder, checkpoints_dir, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir, 'images'))
            os.makedirs(os.path.join(out_dir, 'stats'))

        self.model_name = model_name
        self.test_folder = test_folder
        self.checkpoints_dir = checkpoints_dir
        self.out_dir = out_dir

class SegAccuracy(object):
    # Image to image translation based models evaluation.
    def __init__(self, model_name, segModel, direction, test_folder, checkpoints_dir, segmentation_path, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(os.path.join(out_dir, 'images'))
            os.makedirs(os.path.join(out_dir, 'stats'))

        self.model_name = model_name
        self.segModel = segModel
        self.direction = direction
        self.checkpoints_dir = checkpoints_dir
        self.segmentation_path = segmentation_path
        self.out_dir = out_dir
        self.test_folder = test_folder
        
        if model_name == 'sasan':
            self.model_generator = self.get_sasan_generator()
        elif model_name == 'cyclegan':
            self.model_generator = self.get_cyclegan_generator()
        elif model_name == 'ugatit':
            self.model_generator = self.get_ugatit_generator()

        if direction == 'mr2ct':
            self.segmentor = self.get_segmentor('mr')
        else:
            self.segmentor = self.get_segmentor('ct')

        if model_name == 'sasan':
            self.img_generation = self.generate_img_sasan
        elif model_name == 'cyclegan':
            self.img_generation = self.generate_img_cycle
        elif model_name == 'ugatit':
            self.img_generation = self.generate_img_ugatit

    def get_segmentor(self, modality):
        if self.segModel=="UNET++":
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
            dropout_op_kwargs = {'p': 0, 'inplace': True}
            net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            model = Generic_UNet(INPUT_CH, 64, N_CLASSES,
                                    4,
                                    num_conv_per_stage=2,
                                    norm_op=nn.InstanceNorm2d, 
                                    norm_op_kwargs=norm_op_kwargs,
                                    dropout_op_kwargs=dropout_op_kwargs,
                                    nonlin_kwargs=net_nonlin_kwargs,
                                    final_nonlin=lambda x: x,
                                    convolutional_pooling=True, 
                                    convolutional_upsampling=True).cuda()
        else:
            model = SegmentationModel(INPUT_CH, N_CLASSES).cuda()
        print(self.segmentation_path)
        model = load_network(model, modality, 'latest', self.segmentation_path)
        model.eval()
        return model
    
    def read_nii(self, path):
        return medio.load(path)[0]

    def process_img_other(self, vol):
        batch_wise = self.process_img(vol)
        for i in range(batch_wise.shape[0]):
            batch_wise[i] = (batch_wise[i] - batch_wise[i].min())/(batch_wise[i].max() - batch_wise[i].min())
        
        return batch_wise

    def process_img(self, vol):
        vol = np.flip(vol, axis=0)
        vol = np.flip(vol, axis=1)

        if self.direction == 'mr2ct':
            param1 = -2.8
            param2 = 3.2
        else:
            param1 = -1.8
            param2 = 4.4
        
        batch_wise = np.transpose(vol, (2, 0, 1))

        batch_wise = 2*(batch_wise - param1)/(param2 - param1) - 1.0
        return batch_wise
    
    def process_seg(self, vol):
        vol = np.flip(vol, axis=0)
        vol = np.flip(vol, axis=1)
        batch_wise = np.transpose(vol, (2, 0, 1))
        return batch_wise
    
    def get_sasan_generator(self):
        model = GeneratorED(INPUT_CH, N_ATTENTIONS).cuda()
        attention = UnetAttention(INPUT_CH, 5, N_ATTENTIONS).cuda()

        if self.direction == 'mr2ct':
            model = load_network(model, 'G_A', 'latest', self.checkpoints_dir)
            attention = load_network(attention, 'Attention_B', 'latest', self.checkpoints_dir)
        else:
            model = load_network(model, 'G_B', 'latest', self.checkpoints_dir)
            attention = load_network(attention, 'Attention_A', 'latest', self.checkpoints_dir)
        
        model.eval()
        attention.eval()

        return model, attention

    def get_cyclegan_generator(self):
        model = CycleGANGenerator(3, 3).cuda()
        if self.direction == 'mr2ct':
            model = load_network(model, 'G_A', 'latest', self.checkpoints_dir)
        else:
            model = load_network(model, 'G_B', 'latest', self.checkpoints_dir)
        
        model.eval()
        return model

    def get_ugatit_generator(self):
        def load(net, path, modality):
            params = torch.load(path)
            net.load_state_dict(params[modality])
            return net
        
        model = UGATITResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=4, img_size=256, light=True).cuda()

        if self.direction == 'mr2ct':
            model = load(model, os.path.join(self.checkpoints_dir, 'model', 'mr_ct_params_latest.pt'), 'genB2A')
        else:
            model = load(model, os.path.join(self.checkpoints_dir, 'model', 'mr_ct_params_latest.pt'), 'genA2B')

        model.eval()
        return model

    def generate_img_sasan(self, img):
        G = self.model_generator[0]
        A = self.model_generator[1]
    
        return G(img, A(img)[0])[0].detach()

    def generate_img_cycle(self, img):
        return self.model_generator(img)[0].detach()

    def generate_img_ugatit(self, img):
        return self.model_generator(img)[0][0].detach()

    def normalize_img(self, img):
        return (img - img.min())/(img.max() - img.min())
    
    def predict_seg(self):
        if not os.path.exists(os.path.join(self.out_dir, 'sasan_prediction', self.direction)):
            os.makedirs(os.path.join(self.out_dir, 'sasan_prediction', self.direction))
            os.makedirs(os.path.join(self.out_dir, 'ground_truth', self.direction))
        
        all_img_paths = sorted([f for f in os.listdir(os.path.join(self.test_folder, 'images')) if f.endswith('.nii.gz')])
        all_seg_paths = sorted([f for f in os.listdir(os.path.join(self.test_folder, 'labels')) if f.endswith('.nii.gz')])

        n_samples = len(all_img_paths)
        assert n_samples == len(all_seg_paths)

        for i in range(n_samples):
            img_vol = self.process_img(self.read_nii(os.path.join(self.test_folder, 'images', all_img_paths[i])))
            
            seg_vol = self.process_seg(self.read_nii(os.path.join(self.test_folder, 'labels', all_seg_paths[i])))
            
            prediction_vol = np.zeros_like(seg_vol)
            
            for j in range(1, len(img_vol)-1):
                input_img = torch.from_numpy(img_vol[(j-1):(j+2)].copy()).to(torch.float32)
                input_img = input_img[None, ...]
                input_img = input_img.cuda()
                predicted_img = self.img_generation(input_img)
                predicted_seg = self.segmentor(predicted_img[None, ...])
                if self.segModel=="UNET++":
                    predicted_seg = predicted_seg[0]
                predicted_seg = torch.argmax(predicted_seg, dim=1)

                # convert to numpy
                predicted_seg = predicted_seg[0].cpu().numpy()
                prediction_vol[j] = predicted_seg.copy()
            
            np.save(os.path.join(self.out_dir, 'sasan_prediction', self.direction, all_img_paths[i][:-7]), prediction_vol)
            np.save(os.path.join(self.out_dir, 'ground_truth', self.direction, all_seg_paths[i][:-7]), seg_vol)
    
    def run_eval(self):
        all_img_paths = sorted([f for f in os.listdir(os.path.join(self.test_folder, 'images')) if f.endswith('.nii.gz')])
        all_seg_paths = sorted([f for f in os.listdir(os.path.join(self.test_folder, 'labels')) if f.endswith('.nii.gz')])

        n_samples = len(all_img_paths)
        assert n_samples == len(all_seg_paths)

        evalution_dice  = np.zeros((n_samples, 4))
        evaluation_assd = np.zeros((n_samples, 4))

        for i in range(n_samples):
            print('evaluating ', i)

            img_vol = self.process_img(self.read_nii(os.path.join(self.test_folder, 'images', all_img_paths[i])))
            
            seg_vol = self.process_seg(self.read_nii(os.path.join(self.test_folder, 'labels', all_seg_paths[i])))

            prediction_vol = np.zeros_like(seg_vol)
            for j in range(1, len(img_vol)-1):
                input_img = torch.from_numpy(img_vol[(j-1):(j+2)].copy()).to(torch.float32)
                input_img = input_img[None, ...]
                input_img = input_img.cuda()
                predicted_img = self.img_generation(input_img)
                predicted_seg = self.segmentor(predicted_img[None, ...])
                if self.segModel=="UNET++":
                    predicted_seg = predicted_seg[0]
                predicted_seg = torch.argmax(predicted_seg, dim=1)

                # convert to numpy
                predicted_seg = predicted_seg[0].cpu().numpy()
                prediction_vol[j] = predicted_seg.copy()

                #save predictions
                gt_seg_img = overlay_seg_img(255*self.normalize_img(img_vol[j]), seg_vol[j])
                pred_seg_img = overlay_seg_img(255*self.normalize_img(predicted_img[1].cpu().numpy()), predicted_seg)

                # input_img
                viz_input = 255*self.normalize_img(img_vol[j])

                Image.fromarray(viz_input.astype(np.uint8)).save(os.path.join(self.out_dir, 'images', str(i) + '_' + str(j) + 'input.png'))
                Image.fromarray(gt_seg_img.astype(np.uint8)).save(os.path.join(self.out_dir, 'images', str(i) + '_' + str(j) + 'gt.png'))
                Image.fromarray(pred_seg_img.astype(np.uint8)).save(os.path.join(self.out_dir, 'images', str(i) + '_' + str(j) + 'pred.png'))
                Image.fromarray((255*self.normalize_img(predicted_img[1].cpu().numpy())).astype(np.uint8)).save(os.path.join(self.out_dir, 'images', str(i) + '_' + str(j) + 'pred_img.png'))
            
            for j in range(1, N_CLASSES):
                seg_vol_class = seg_vol == j
                pred_vol_class = prediction_vol == j

                dice = metric.binary.dc(pred_vol_class, seg_vol_class)
                assd = metric.binary.assd(pred_vol_class, seg_vol_class)

                evalution_dice[i, j-1] = dice
                evaluation_assd[i, j-1] = assd
       
       
        print(self.direction, ' => Results Dice mean = ', evalution_dice.mean())
        print(evalution_dice)
        print(self.direction, '=> Resutls ASSD = ', evaluation_assd.mean())
        print(evaluation_assd)

        print(self.direction, ' => Results Dice mean = ', evalution_dice.mean(axis=0))
        print(self.direction, ' => Results Dice std = ', evalution_dice.std(axis=0))
        print(self.direction, '=> Overall mean = ', evalution_dice.mean(axis=1).mean())
        print(self.direction, '=> Overall std  = ', evalution_dice.mean(axis=1).std())
        print(evalution_dice)

        print(self.direction, ' => Results ASSD mean = ', evaluation_assd.mean(axis=0))
        print(self.direction, ' => Results ASSD std = ', evaluation_assd.std(axis=0))
        print(self.direction, '=> Overall mean = ', evaluation_assd.mean(axis=1).mean())
        print(self.direction, '=> Overall std  = ', evaluation_assd.mean(axis=1).std())
        print(evaluation_assd)

        np.save(os.path.join(self.out_dir, 'stats', self.direction + '_dice'), evalution_dice)
        np.save(os.path.join(self.out_dir, 'stats', self.direction + '_assd'), evalution_dice)
