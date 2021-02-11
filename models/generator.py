from .architectures import ResnetBlock, SPADEResnetBlock, ConvTransposeBlock, ConvBlock
import torch.nn as nn
from .attention import AttentionModel

import torch.nn.functional as F
import torch

class Generator(nn.Module): 
    """Generator based on Resnet, takes 256x256 input"""
    def __init__(self, nch_input, nch_output, n_attentions, nfilters=64):
        super(Generator, self).__init__()
        
        self.init_layers = nn.Sequential(ConvBlock(nch_input, nfilters, kernel_size=7, stride=1, padding=3),
                                         ConvBlock(nfilters, nfilters, kernel_size=5, stride=2, padding=2),
                                         ConvBlock(nfilters, 2*nfilters, kernel_size=5, stride=2, padding=2))

        spade_blocks = []
        for i in range(6):
            spade_blocks += [SPADEResnetBlock(2*nfilters, 2*nfilters, n_attentions)]
        
        self.spade_blocks = nn.ModuleList(spade_blocks)

        self.final_layers = nn.ModuleList([ConvBlock(2*nfilters, nfilters, kernel_size=3, stride=1, padding=1),
                                          ConvBlock(nfilters, nfilters, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(nfilters, nch_output, kernel_size=3, stride=1, padding=1)])

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
    
    def forward(self, input_image, attention_maps):
        """Forward pass. layers has 15 modules"""
        x = self.init_layers(input_image)

        for block in self.spade_blocks:
            x = block(x, attention_maps)

        for i, block in enumerate(self.final_layers):
            x = block(x)
            if i != 2:
                x = self.up(x)
        
        return x

def getencoder(input_channel, encoder_nf):
    model = []
    model += [ConvBlock(input_channel, 32, kernel_size=7, stride=1, padding=3)]
    prev_nf = 32
    for i, nf in enumerate(encoder_nf):
        if i < 2:
            stride = 2
        else:
            stride = 1
        model += [ConvBlock(prev_nf, nf, kernel_size=3, stride=stride, padding=1)]
        prev_nf = nf

    return nn.Sequential(*model)

def getdecoder(decoder_nf, prev_nf, n_attentions):
    model = nn.ModuleList()
    n_spade_levels = 3
    for i in range(n_spade_levels):
        model.append(SPADEResnetBlock(prev_nf, decoder_nf[i], n_attentions))
        prev_nf = decoder_nf[i]

    for i in range(n_spade_levels, len(decoder_nf) - 1):
        model.append(ConvBlock(prev_nf, decoder_nf[i], kernel_size=3, stride=1, padding=1))
        prev_nf = decoder_nf[i]

    model.append(nn.Conv2d(prev_nf, decoder_nf[-1], kernel_size=3, stride=1, padding=1))

    return model


class GeneratorED(nn.Module):
    """
    Based on Encoder Decoder architecture. Spade included only in the decoder part.
    """

    def __init__(self, input_channels, n_attentions, encoder_nf=[64, 128, 128], decoder_nff=[128, 64, 32, 16]):
        decoder_nf = decoder_nff.copy()
        decoder_nf.append(input_channels)
        super(GeneratorED, self).__init__()
        self.encoder = getencoder(input_channels, encoder_nf)
        self.decoder = getdecoder(decoder_nf, 128, n_attentions)
        self.up      = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x, attention):
        # encode the image
        en_out = self.encoder(x)
        x = en_out

        # spade normlayers
        for i in range(2):
            x = self.decoder[i](x, attention)
            x = self.up(x)

        x = self.decoder[2](x, attention)
        # normal convolutions
        for i in range(3, len(self.decoder)):
            x = self.decoder[i](x)

        return torch.tanh(x), en_out