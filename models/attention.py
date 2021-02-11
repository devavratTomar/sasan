import torch
import torch.nn as nn
import numbers
import math

from .architectures import ResnetBlock, ConvBlock, ConvTransposeBlock

import torch.nn.functional as F

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


class AttentionModel(nn.Module): 
    '''ResNet-based generator for attention mask prediction.'''
    def __init__(self, in_nc, ngf=32, natt=32, nclasses=5):
        super(AttentionModel, self).__init__()
        
        self.natt = natt

        self.backbone_1 = ConvBlock(in_nc, ngf, kernel_size=7, stride=2, padding=3)
        self.backbone_2 = ConvBlock(ngf, 2*ngf, kernel_size=5, stride=2, padding=2)

        self.backbone_3 = nn.Sequential(ResnetBlock(2*ngf),
                                        ResnetBlock(2*ngf),
                                        ResnetBlock(2*ngf))

        self.atten_layer = nn.Conv2d(2*ngf, natt, kernel_size=3, stride=1, padding=1)
        
        self.class_layer_1 = ConvBlock(2*ngf, nclasses, kernel_size=5, stride=1, padding=2)
        self.class_layer_2 = nn.Conv2d(nclasses, nclasses, kernel_size=5, stride=1, padding=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')


        # self.class_layer = nn.Sequential(ConvTransposeBlock(2*ngf, ngf, kernel_size=4, stride=2, padding=1),
        #                                  ConvTransposeBlock(ngf, ngf, kernel_size=4, stride=2, padding=1),
        #                                  nn.Conv2d(ngf, nclasses, kernel_size=7, stride=1, padding=3))
    
    def forward(self, x):
        x_1 = self.backbone_1(x)
        x_2 = self.backbone_2(x_1)
        x_3 = self.backbone_3(x_2)

        attentions = self.atten_layer(x_3)

        class_1 = self.class_layer_1(self.up(x_3))
        logits = self.class_layer_2(self.up(class_1))

        return attentions, logits



# class UnetAttention(nn.Module):
#     def __init__(self, input_channels, n_classes, n_attentions, encoder=[64, 64, 128, 256], decoder=[256, 128, 64, 64]):
#         super(UnetAttention, self).__init__()
#         self.enc_nf, self.dec_nf = encoder, decoder
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

#         self.init = ConvBlock(input_channels, 32, kernel_size=5, stride=1, padding=2)

#         # configure encoder
#         prev_nf = 32 # initial number of channels
#         self.downarm = nn.ModuleList()

#         for nf in self.enc_nf:
#             self.downarm.append(ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2))
#             prev_nf = nf

#         enc_history = list(reversed(self.enc_nf))
#         self.uparm = nn.ModuleList()

#         for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
#             channels = prev_nf + enc_history[i] if i > 0 else prev_nf
#             self.uparm.append(ConvBlock(channels, nf, kernel_size=3, stride=1, padding=1))
#             prev_nf = nf

#         prev_nf += 32
#         # final layers
#         self.final = ConvBlock(prev_nf, 32, kernel_size=3, stride=1, padding=1)
#         self.atten_layer = nn.Conv2d(32, n_attentions, kernel_size=3, stride=1, padding=1, bias=False) #kernel_size 5
#         self.logit_layer = nn.Conv2d(n_attentions, n_classes, kernel_size=3, stride=1, padding=1) # kernel_size 1

#     def forward(self, x):
#         # get encoder activations
#         x_enc = [self.init(x)]
#         for layer in self.downarm:
#             x_enc.append(layer(x_enc[-1]))

#          # conv, upsample, concatenate series
#         x = x_enc.pop()
#         for layer in self.uparm:
#             x = layer(x)
#             x = self.upsample(x)
#             x = torch.cat([x, x_enc.pop()], dim=1)
        
#         x = self.final(x)

#         attentions = self.atten_layer(x)
        
#         logits = self.logit_layer(attentions)

#         return attentions, logits

class UnetAttention(nn.Module):
    def __init__(self, input_channels, n_classes, n_attentions, encoder=[64, 64, 128, 256], decoder=[256, 128, 64, 64]):
        super(UnetAttention, self).__init__()
        self.enc_nf, self.dec_nf = encoder, decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.init = ConvBlock(input_channels, 32, kernel_size=5, stride=1, padding=2)

        # configure encoder
        prev_nf = 32 # initial number of channels
        self.downarm = nn.ModuleList()

        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(prev_nf, nf, kernel_size=5, stride=2, padding=2))
            prev_nf = nf

        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()

        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(channels, nf, kernel_size=3, stride=1, padding=1))
            prev_nf = nf

        prev_nf += 32
        # final layers
        self.final = ConvBlock(prev_nf, 32, kernel_size=3, stride=1, padding=1)
        self.atten_layer = nn.Conv2d(32, n_attentions, kernel_size=3, stride=1, padding=1)
        self.logit_layer = nn.Conv2d(n_attentions, n_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # get encoder activations
        x_enc = [self.init(x)]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

         # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)
        
        x = self.final(x)

        attentions = self.atten_layer(x)
        
        logits = self.logit_layer(attentions)

        #attentions = attentions/(torch.sqrt((attentions**2).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)) + 1e-6)

        return attentions, logits#F.softmax(attentions, dim=1), logits