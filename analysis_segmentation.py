import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import utilities.util as util

from dataset import WholeHeartDatasetPostProcessed

from models import SegmentationModel
from models.generator import Generator
from models.attention import AttentionModel

from PIL import Image
import os

import numpy as np

from medpy import metric

import json

colors = np.array([
    [0,     0,   0],
    [254, 232,  81], #LV-myo
    [145, 193,  62], #LA-blood
    [ 29, 162, 220], #LV-blood
    [238,  37,  36]]) #AA

RESCALE = True
TEST    = True

def mean_filtered(array, lower, higher):
    filtered = np.array([item for item in array if item>=lower and item<=higher])
    return filtered.mean()

def std_filtered(array, lower=5, higher=5):
    filtered = np.array([item for item in array if item>=lower and item<=higher])
    return filtered.std()

def overlay_seg_img(img, seg):
    # get unique labels
    labels = np.unique(seg)

    # remove background
    labels = labels[labels !=0]

    # img backgournd
    img_b = img*(seg == 0)

    # final_image
    final_img = np.zeros([img.shape[0], img.shape[1], 3])

    final_img += img_b[:, :, np.newaxis]

    for l in labels:
        mask = seg == l
        img_f = img*mask

        # convert to rgb
        img_f = np.tile(img_f, (3, 1, 1)).transpose(1, 2, 0)

        # colored segmentation
        img_seg = colors[l*mask]

        # alpha overlay
        final_img += 0.5*img_f + 0.5*img_seg
    
    return final_img

def original_segmentation_results(modality, data_root_dir, model_checkpoint_path, output_path,  use_gpu=True, batch_size=16):
    seg_model = SegmentationModel(3, 5)
    seg_model = util.load_network(seg_model, modality, 'latest', model_checkpoint_path)
    seg_model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if use_gpu:
        seg_model = seg_model.cuda()

    dataset = WholeHeartDatasetPostProcessed(data_root_dir, test=TEST)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=4)

    it = 0
    accuracy = {0: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                1: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                2: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                3: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                4: {'jc':[], 'dc':[], 'assd':[], 'asd':[]}}

    for img, seg in dataloader:
        if use_gpu:
            img = img.cuda()
        
        predictions = seg_model(img).detach()
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()
        img = img.cpu().numpy()
        seg = seg.cpu().numpy()
        
        # predictions are now of shape NxHxW
        for i in range(img.shape[0]):
            single_img = img[i][0] # one channel image
            single_seg = seg[i]    # segentations are of the form NxHxW
            single_pred = predictions[i] # predictions are of the form NxHxW

            # normalizations for better visualization

            single_img_viz  = 255*(single_img - single_img.min())/(single_img.max() - single_img.min() + 1e-6)
            single_seg_viz  = overlay_seg_img(single_img_viz, single_seg)
            single_pred_viz = overlay_seg_img(single_img_viz, single_pred)

            single_img_viz  = np.tile(single_img_viz, (3, 1, 1)).transpose(1, 2, 0)

            output_img = np.concatenate([single_img_viz, single_seg_viz, single_pred_viz], axis=1).astype(np.uint8)
            Image.fromarray(output_img).save(os.path.join(output_path, 'orig_seg_out_' + str(it).zfill(6) + '.png'))

            stats = ['orig_seg_out_'+ str(it).zfill(6), 'na', 'na', 'na', 'na', 'na']
            # numerical accuracy of segmentation for 5 different classes
            for c in range(5):
                if c in single_seg:# and c in single_pred:
                    gt_mask = (single_seg == c)
                    pred_mask = (single_pred == c)
                    jc = metric.binary.jc(pred_mask, gt_mask)
                    dc  = metric.binary.dc(pred_mask, gt_mask)
                    if c in single_pred:
                        assd = metric.binary.assd(pred_mask, gt_mask)
                        asd  = metric.binary.asd(pred_mask, gt_mask)
                        accuracy[c]['assd'].append(assd)
                        accuracy[c]['asd'].append(asd)
                    else:
                        assd = 50
                        asd  = 50
                        accuracy[c]['assd'].append(assd)
                        accuracy[c]['asd'].append(asd)
                    
                    accuracy[c]['jc'].append(jc)
                    accuracy[c]['dc'].append(dc)

                    stats[c+1] = [('jc', jc), ('dc', dc), ('assd', assd), ('asd', asd)]
            
            it += 1

            with open(os.path.join(output_path, 'output_stats.txt'), 'a') as f:
                f.write(json.dumps(stats) + '\n')
    
    # overall
    overall_accuracy= {'jc':[], 'dc':[], 'assd':[], 'asd':[]}
    
    for c in range(1, 5):
        overall_accuracy['jc'] += accuracy[c]['jc']
        overall_accuracy['dc'] += accuracy[c]['dc']
        overall_accuracy['assd'] += accuracy[c]['assd']
        overall_accuracy['asd'] += accuracy[c]['asd']
    
    for c in range(5):
        accuracy[c] = [['jc', np.mean(accuracy[c]['jc']), np.std(accuracy[c]['jc'])],
                       ['dc', np.mean(accuracy[c]['dc']), np.std(accuracy[c]['dc'])],
                       ['assd', np.mean(accuracy[c]['assd']), np.std(accuracy[c]['assd'])],
                       ['asd', np.mean(accuracy[c]['asd']), np.std(accuracy[c]['asd'])]]

    overall_accuracy['jc'] = [np.mean(overall_accuracy['jc']), np.std(overall_accuracy['jc'])]
    overall_accuracy['dc'] = [np.mean(overall_accuracy['dc']), np.std(overall_accuracy['dc'])]
    overall_accuracy['assd'] = [np.mean(overall_accuracy['assd']), np.std(overall_accuracy['assd'])]
    overall_accuracy['asd'] = [np.mean(overall_accuracy['asd']), np.std(overall_accuracy['asd'])]
    
    with open(os.path.join(output_path, 'fake_output_stats.txt'), 'a') as f:
        f.write('Overall average per class\n')
        f.write(json.dumps(accuracy) + '\n')
        f.write(json.dumps(overall_accuracy)+ '\n')
    
    print(json.dumps(accuracy))
    print(json.dumps(overall_accuracy))

def generate_fake_images(data_root, modality, checkpoint_path, output_path, use_gpu=True, batch_size=16):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    net = Generator(3, 3, 16)
    attention = AttentionModel(3, 64, 16, 5)

    if modality == 'mr':
        net = util.load_network(net, 'G_A', 'latest', checkpoint_path)
        attention = util.load_network(attention, 'Attention_B', 'latest', checkpoint_path)
    else:
        net = util.load_network(net, 'G_B', 'latest', checkpoint_path)
        attention = util.load_network(attention, 'Attention_A', 'latest', checkpoint_path)

    if use_gpu:
        net = net.cuda()
        attention = attention.cuda()
    
    net.eval()
    attention.eval()

    dataset = WholeHeartDatasetPostProcessed(data_root, True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=4)

    it = 0
    for x, _ in dataloader:
        if use_gpu:
            x = x.cuda()
        
        atten, _ = attention(x)
        atten = atten.detach()
        fake_mr_imgs = net(x, atten).detach().cpu().numpy()
        x = x.cpu().numpy()
        atten = F.softmax(atten, dim=1)
        atten = atten.cpu().numpy()

        for i in range(fake_mr_imgs.shape[0]):
            img = fake_mr_imgs[i][0]
            in_img = x[i][0]

            img = 255*(img - img.min())/(img.max() - img.min() + 1e-6)
            in_img = 255*(in_img - in_img.min())/(in_img.max() - in_img.min() + 1e-6)
            
            out = np.concatenate([in_img, img], axis=1).astype(np.uint8)
            out_atten = 255*np.concatenate(atten[i], axis=1)
            out_atten = out_atten.astype(np.uint8)

            Image.fromarray(out).save(os.path.join(output_path, str(it).zfill(6) + '.png'))
            Image.fromarray(out_atten).save(os.path.join(output_path, str(it).zfill(6) + '_atten.png'))
            it = it + 1


def domain_shift_segmentation_results(modality, data_root, seg_model_checkpoint, img_tr_model_checkpoint, output_path,  use_gpu=True, batch_size=8):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    seg_model = SegmentationModel(3, 5)
    generator = Generator(3, 3, 16)
    attention = AttentionModel(3, 64, 16, 5)

    gen_str = 'G_A' if modality == 'mr' else 'G_B'
    atten_str = 'Attention_B' if modality == 'mr' else 'Attention_A'

    seg_model = util.load_network(seg_model, modality, 'latest', seg_model_checkpoint)
    generator = util.load_network(generator, gen_str, 'latest', img_tr_model_checkpoint)
    attention = util.load_network(attention, atten_str, 'latest', img_tr_model_checkpoint)

    seg_model.eval()
    generator.eval()
    attention.eval()

    if use_gpu:
        seg_model = seg_model.cuda()
        generator = generator.cuda()
        attention = attention.cuda()

    dataset = WholeHeartDatasetPostProcessed(data_root,  test=TEST)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=4)

    it = 0
    accuracy = {0: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                1: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                2: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                3: {'jc':[], 'dc':[], 'assd':[], 'asd':[]},
                4: {'jc':[], 'dc':[], 'assd':[], 'asd':[]}}

    for x, seg in dataloader:
        if use_gpu:
            x = x.cuda()

        atten, _ = attention(x)
        atten = atten.detach()

        # fake mr
        fake_mr = generator(x, atten).detach()

        # get the segmenation label for fake mr using the segmentation model
        fake_pred = seg_model(fake_mr).detach()
        fake_pred = torch.argmax(fake_pred, dim=1).cpu().numpy()

        # without domain adaptation (directly feed ct image)
        ct_no_adaptation_pred = seg_model(x).detach()
        ct_no_adaptation_pred = torch.argmax(ct_no_adaptation_pred, dim=1).cpu().numpy()

        x = x.cpu().numpy()
        seg = seg.cpu().numpy()
        fake_mr = fake_mr.cpu().numpy()

        # now analysis for all the test ct images
        for i in range(x.shape[0]):
            if seg[i].sum() == 0:
                continue
            single_ct = x[i][0]
            single_seg = seg[i]
            single_fake_mr = fake_mr[i][0]
            single_fake_pred = fake_pred[i]
            single_ct_no_adaptation_pred = ct_no_adaptation_pred[i]

            # normalizations for better visualization
            single_ct_viz  = 255*(single_ct - single_ct.min())/(single_ct.max() - single_ct.min() + 1e-6)
            single_seg_viz  = overlay_seg_img(single_ct_viz, single_seg)

            single_fake_mr_viz = 255*(single_fake_mr - single_fake_mr.min())/(single_fake_mr.max() - single_fake_mr.min() + 1e-6)
            single_fake_pred_viz = overlay_seg_img(single_fake_mr_viz, single_fake_pred)

            # wihtout domain adaptaiton
            single_ct_no_adaptation_pred = overlay_seg_img(single_ct_viz, single_ct_no_adaptation_pred)
            
            
            single_ct_viz = np.tile(single_ct_viz, (3, 1, 1)).transpose(1, 2, 0)
            single_fake_mr_viz = np.tile(single_fake_mr_viz, (3, 1, 1)).transpose(1, 2, 0)

            # original ct, synthesized mri, segmentation without adaptaion, segmentation result domain adaptation, ground truth
            output_img = np.concatenate([single_ct_viz, single_fake_mr_viz, single_ct_no_adaptation_pred, single_fake_pred_viz, single_seg_viz], axis=1).astype(np.uint8)
            Image.fromarray(output_img).save(os.path.join(output_path, 'fake_seg_out_' + str(it).zfill(6) + '.png'))
            
            stats = ['fake_seg_out_'+ str(it).zfill(6), 'na', 'na', 'na', 'na', 'na']
            
            # numerical accuracy of segmentation for 5 different classes
            for c in range(5):
                if c in single_seg:
                    gt_mask = (single_seg == c)
                    pred_mask = (single_fake_pred == c)

                    jc = metric.binary.jc(pred_mask, gt_mask)
                    dc = metric.binary.dc(pred_mask, gt_mask)
                    if c in single_fake_pred:
                        assd = metric.binary.assd(pred_mask, gt_mask)
                        asd  = metric.binary.asd(pred_mask, gt_mask)
                        accuracy[c]['assd'].append(assd)
                        accuracy[c]['asd'].append(asd)
                    else:
                        assd = -1
                        asd  = -1

                    accuracy[c]['jc'].append(jc)
                    accuracy[c]['dc'].append(dc)

                    stats[c+1] = [('jc', jc), ('dc', dc), ('assd', assd), ('asd', asd)]
            
            it += 1

            with open(os.path.join(output_path, 'fake_output_stats.txt'), 'a') as f:
                f.write(json.dumps(stats) + '\n')
    
    # overall
    overall_accuracy= {'jc':[], 'dc':[], 'assd':[], 'asd':[]}
    
    for c in range(1, 5):
        overall_accuracy['jc'] += accuracy[c]['jc']
        overall_accuracy['dc'] += accuracy[c]['dc']
        overall_accuracy['assd'] += accuracy[c]['assd']
        overall_accuracy['asd'] += accuracy[c]['asd']
    
    for c in range(5):
        accuracy[c] = [['jc', len(accuracy[c]['jc']), np.mean(accuracy[c]['jc']), np.std(accuracy[c]['jc'])],
                       ['dc', len(accuracy[c]['dc']), np.mean(accuracy[c]['dc']), np.std(accuracy[c]['dc'])],
                       ['assd', len(accuracy[c]['assd']), np.mean(accuracy[c]['assd']), np.std(accuracy[c]['assd'])],
                       ['asd', len(accuracy[c]['asd']), np.mean(accuracy[c]['asd']), np.std(accuracy[c]['asd'])]]

    overall_accuracy['jc'] = [np.mean(overall_accuracy['jc']), np.std(overall_accuracy['jc'])]
    overall_accuracy['dc'] = [np.mean(overall_accuracy['dc']), np.std(overall_accuracy['dc'])]
    overall_accuracy['assd'] = [np.mean(overall_accuracy['assd']), np.std(overall_accuracy['assd'])]
    overall_accuracy['asd'] = [np.mean(overall_accuracy['asd']), np.std(overall_accuracy['asd'])]
    
    with open(os.path.join(output_path, 'fake_output_stats.txt'), 'a') as f:
        f.write('Overall average per class\n')
        f.write(json.dumps(accuracy) + '\n')
        f.write(json.dumps(overall_accuracy)+ '\n')
    
    print(json.dumps(accuracy))
    print(json.dumps(overall_accuracy))


def generate_fake_data_training(modality, model_path, input_path, output_path, use_gpu=True, batch_size=20):
    if not os.path.exists(output_path):
        os.makedirs(os.path.join(output_path, 'mr' + '_train', 'images'))
        os.makedirs(os.path.join(output_path, 'mr' + '_train', 'labels'))
        os.makedirs(os.path.join(output_path, 'ct' + '_train', 'images'))
        os.makedirs(os.path.join(output_path, 'ct' + '_train', 'labels'))

    generator = Generator(3, 3, 16)
    attention = AttentionModel(3, 64, 16, 5)

    gen_str = 'G_A' if modality == 'mr' else 'G_B'
    atten_str = 'Attention_B' if modality == 'mr' else 'Attention_A'

    generator = util.load_network(generator, gen_str, 'latest', model_path)
    attention = util.load_network(attention, atten_str, 'latest', model_path)

    generator.eval()
    attention.eval()

    if use_gpu:
        generator = generator.cuda()
        attention = attention.cuda()

    dataset = WholeHeartDatasetPostProcessed(input_path,  test=TEST)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=False, num_workers=4)
    it = 0
    for x, seg in dataloader:
        if use_gpu:
            x = x.cuda()
        
        attention_maps = attention(x)[0].detach()
        fake_imgs = generator(x, attention_maps).detach()

        for j in range(fake_imgs.shape[0]):
            img_numpy = fake_imgs[j][0].cpu().numpy()
            seg_numpy = seg[j].cpu().numpy()
            np.save(os.path.join(output_path, modality + '_train', 'images', 'fake_' + str(it).zfill(6)), img_numpy)
            np.save(os.path.join(output_path, modality + '_train', 'labels', 'fake_' + str(it).zfill(6)), seg_numpy)
            it += 1

    
if __name__ == '__main__':
    dataroot_mr = '../../img_modality_datasets/whole_heart/mr_test/'
    dataroot_ct = '../../img_modality_datasets/whole_heart/ct_test/'
    dataroot_ct_train = '../../img_modality_datasets/whole_heart/ct_train/'

    # original_segmentation_results('mr', dataroot_mr, './checkpoints_segmentation_mr_rescale', 'original_mr_seg_results_rescale')
    # original_segmentation_results('ct', dataroot_ct, './checkpoints_segmentation_ct_rescale', 'original_ct_seg_results_rescale')
    # domain_shift_mr_segmentation_results(dataroot_ct, './checkpoints_segmentation_mr_rescale', './checkpoints_wholeheart_cyclesegment_orthonormal_attentions_dice_2discriminator_rescaled_images', 'fake_mr_seg_results_2_dis_rescaled')
    domain_shift_segmentation_results('ct', dataroot_mr, './checkpoints_segmentation_ct', './checkpoints_domain_adapt', 'fake_ct_seg_results')
    # generate_fake_images(dataroot_ct, 'mr', './checkpoints_wholeheart_cyclesegment_ortho_attentions_id_dice_reeduced_discriminator', 'fake_mr_images_dice')
    # generate_fake_images(dataroot_mr, 'ct', './checkpoints_wholeheart_cyclesegment_ortho_attentions_id_dice_reeduced_discriminator', 'fake_ct_images_dice')

    # lower bounds
    # original_segmentation_results('mr', dataroot_ct, './checkpoints_segmentation_mr_rescale', 'segmentation_ct_without_domain_adaptation_rescale')

    # generate fake data
    #generate_fake_data_training('mr',
    #                            './checkpoints_wholeheart_cyclesegment_orthonormal_attentions_dice_2discriminator_rescaled_images',
    #                            dataroot_ct_train,
    #                            '../../img_modality_datasets/whole_heart/clean_crop_fake')

   
    #original_segmentation_results('mr', dataroot_mr, './checkpoints_segmentation_mr_data_aug_rescale', 'original_mr_seg_results__aug_rescale')