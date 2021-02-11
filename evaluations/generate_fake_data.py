import torch
from utilities.util import load_network

import torch
import torch.nn as nn
import os

import numpy as np
from models import UnetAttention, GeneratorED

INPUT_CH = 3
N_ATTENTIONS = 8
N_CLASSES = 5

@torch.no_grad()
def generate_fake_data(modality, input_folder, output_folder, checkpoints_dir):
    model = GeneratorED(INPUT_CH, N_ATTENTIONS).cuda()
    attention = UnetAttention(INPUT_CH, 5, N_ATTENTIONS).cuda()
    
    if modality == 'mr':
        model = load_network(model, 'G_A', 'latest', checkpoints_dir)
        attention = load_network(attention, 'Attention_B', 'latest', checkpoints_dir)
    else:
        model = load_network(model, 'G_B', 'latest', checkpoints_dir)
        attention = load_network(attention, 'Attention_A', 'latest', checkpoints_dir)

    model = model.cuda()
    attention = attention.cuda()
    model.eval()
    attention.eval()

    if not os.path.exists(output_folder):
        os.makedirs(os.path.join(output_folder, 'images'))
        os.makedirs(os.path.join(output_folder, 'labels'))

    all_imgs = [f for f in os.listdir(os.path.join(input_folder, 'images')) if f.endswith('.npy')]

    for i in range(len(all_imgs)):
        img_path = os.path.join(input_folder, 'images', all_imgs[i])

        img = np.load(img_path).transpose(2, 0, 1) # chanenl first
        img = img[None, ...] # add batch
        img = torch.from_numpy(img).to(torch.float32).cuda()

        fake_atten, fake_logits = attention(img)
        fake_img, _ = model(img, fake_atten)
        fake_label = torch.argmax(fake_logits, dim=1)

        fake_img = fake_img.cpu().numpy()[0]
        fake_label = fake_label.cpu().numpy()[0]

        fake_img = fake_img.transpose(1, 2, 0) # back to channel last

        np.save(os.path.join(output_folder, 'images', 'fake_img_'+ str(i)), fake_img)
        np.save(os.path.join(output_folder, 'labels', 'fake_label_' + str(i)), fake_label)
        