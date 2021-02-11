import torch
import torch.nn as nn

import torch.nn.functional as F

class ResnetBlock(nn.Module):
    """Resnet block"""
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)
        
        
    def build_conv_block(self, dim, use_dropout):

        # padd input
        conv_block = [nn.ReflectionPad2d(1)]
     
        # add convolutional layer followed by normalization and ReLU
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0), 
                           nn.InstanceNorm2d(dim), 
                           nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        conv_block += [nn.ReflectionPad2d(1)]
 
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0), 
                       nn.InstanceNorm2d(dim)]
    
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward pass (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class CycleGANGenerator(nn.Module): 
    """Generator based on Resnet, takes 256x256 input"""
    def __init__(self, nch_input, nch_output, nfilters=64, use_dropout=False):
        super(Generator, self).__init__()
        
        # module list to access intermediate outputs

        self.layers = nn.ModuleList()

        # adding a padding followed by a convolutional layer, instance normalization and ReLU
        self.layers.append(nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(nch_input, nfilters, kernel_size=7, padding=0),
                                         nn.InstanceNorm2d(nfilters),
                                         nn.LeakyReLU(0.2, True))
                          )
    
        
        n_strided = 2
        
        # two strided convolutional layers (reduce the size twice in each dimension) on each iteration step

        chanel_size = nfilters
        self.layers.append(nn.Sequential(nn.Conv2d(chanel_size, chanel_size * 2, kernel_size=3, stride=2, padding=1),
                                             nn.InstanceNorm2d(chanel_size * 2),
                                             nn.LeakyReLU(0.2, True))
                              )
        
        chanel_size = 2*nfilters
        self.layers.append(nn.Conv2d(chanel_size, chanel_size * 2, kernel_size=3, stride=2, padding=1))
        
        # nine residual blocks
        for i in range(6):
            self.layers.append(ResnetBlock(4 * nfilters, False))
            
            
        # two fractionally strided convolution layers
        for i in range(n_strided):
            
            chanel_size = 2 ** (n_strided-i) * nfilters

            self.layers.append(nn.Sequential(nn.ConvTranspose2d(chanel_size, int(chanel_size / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                                             nn.InstanceNorm2d(chanel_size * 2),
                                             nn.LeakyReLU(0.2, True))
                              )
                
                
        # last convolutional layer   
        self.layers.append(nn.Sequential(nn.ReflectionPad2d(3),
                                         nn.Conv2d(nfilters, nch_output, kernel_size=7, padding=0),
                                         nn.LeakyReLU(0.2, True),
                                         nn.Conv2d(nch_output, nch_output, kernel_size=1, padding=0))
                          )
    
    def forward(self, input_image):
        """Forward pass. layers has 15 modules"""

        # first layer
        out = self.layers[0](input_image)

        # downsampling layers
        out = self.layers[1](out)
        out = self.layers[2](out)

        # 6 resnet blocks
        out = self.layers[3](out)
        out = self.layers[4](out)
        out = self.layers[5](out)
        out = self.layers[6](out)
        out = self.layers[7](out)
        out = self.layers[8](out)

        # 2 upsampling layers
        out = self.layers[9](out)
        out = self.layers[10](out)

        # final output layer
        out = self.layers[11](out)

        return out