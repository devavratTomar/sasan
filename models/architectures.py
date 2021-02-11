import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()

        self.main = nn.Sequential(nn.ReflectionPad2d(padding),
                                  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=bias))
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ConvTransposeBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True):
        super().__init__()

        self.main = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ResnetBlock(nn.Module):
    """Resnet block"""
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)
        
        
    def build_conv_block(self, dim):

        # padd input
        conv_block = [nn.ReflectionPad2d(1)]
     
        # add convolutional layer followed by normalization and ReLU
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0), 
                       nn.InstanceNorm2d(dim), 
                       nn.LeakyReLU(0.2, True),
                       nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(dim)]
    
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward pass (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class SPADEConv(nn.Module):
    def __init__(self, fin, fout, n_attention, kernel_size, stride, padding):
        super(SPADEConv, self).__init__()

        self.conv = nn.Conv2d(fin, fout, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = SPADE(fin, n_attention)

    def forward(self, x, seg):
        dx = self.conv(self.actvn(self.norm(x, seg)))
        return dx

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, n_attention):
        super(SPADEResnetBlock, self).__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = SPADE(fin, n_attention)
        self.norm_1 = SPADE(fmiddle, n_attention)
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, n_attention)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x, seg):
        x_s = self.shortcut(x, seg)

        dx = self.conv_0(self.actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)

class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc):
        super(SPADE, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 64
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)
        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out