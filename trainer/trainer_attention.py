import torch
import torch.nn as nn
import os
import numbers
import math
from utilities.util import load_network

from models.discriminator import Discriminator, DiscriminatorLight
from models.generator import GeneratorED
from models.attention import UnetAttention

from loss import GANLoss, OrthonormalityLoss, CrossEntropyLossWeighted
import torch.nn.functional as F
import kornia
from pytorch_msssim import SSIM

import random

from utilities.image_pool import ImagePool

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.pad = kernel_size//2

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
        
        
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )


    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        input = F.pad(input, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


class TrainerAttention(object):
    def __init__(self, opt):
        self.opt = opt
        self.attention_model = UnetAttention(opt.input_ch, 5, opt.n_attention)

        if opt.modality == 'mr':
            self.model_name = 'Attention_A'
        else:
            self.model_name = 'Attention_B'
        
        if opt.continue_train:
            self.attention_model = load_network(self.attention_model, self.model_name, 'latest', opt.checkpoints_dir)

        if len(opt.gpu_ids) > 0:
            self.attention_model = self.attention_model.cuda()

        self.criterian_ce  = CrossEntropyLossWeighted()
        self.criterian_dice = kornia.losses.DiceLoss()

        self.criterian_ortho = OrthonormalityLoss(opt.n_attention)

        self.optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=0.0002)


    def run_train_step(self, img, seg):
        self.optimizer.zero_grad()
        attentions, logits = self.attention_model(img)
        loss_ce = self.criterian_ce(logits, seg)
        loss_dice = self.criterian_dice(logits, seg)
        loss_ortho = self.criterian_ortho(attentions)

        loss = loss_ce + 0.2*loss_dice + loss_ortho

        loss.backward()
        self.optimizer.step()

        self.losses = {
            'dice':loss_dice.detach(),
            'ce': loss_ce.detach(),
            'ortho': loss_ortho.detach()
        }

        self.attentions = 2*F.softmax(attentions, dim=1) - 1.0
        self.attentions = self.attentions.detach().view(-1, 1, self.attentions.shape[2], self.attentions.shape[3])
        self.predictions_index = torch.argmax(logits, dim=1, keepdim=True)

    def run_test(self, img):
        _, prediction = self.attention_model(img)
        prediction = prediction.detach()
        return torch.argmax(prediction, dim=1, keepdim=True)

    def save(self, epoch):
        att_name = '%s_net_%s.pth' % (epoch, self.model_name)
        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')
        torch.save(self.attention_model.state_dict(), os.path.join(saved_model_path, att_name))

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
