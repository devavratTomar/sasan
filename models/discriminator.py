import torch.nn as nn

class Discriminator(nn.Module):
    """Discriminator Class"""
    def __init__(self, nch_input, nfilters=32, nlayers=3):
        
        super(Discriminator, self).__init__()
        
        # head
        model = [nn.Conv2d(nch_input, nfilters, kernel_size=4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, True)]
        
        fl = nfilters
        for n in range(1, nlayers):
            fl = 2 * fl
            model += [nn.Conv2d(fl//2, fl, kernel_size=4, stride=2, padding=1),
                     nn.InstanceNorm2d(fl),
                     nn.LeakyReLU(0.2, True)]
        
        
        model += [nn.Conv2d(fl, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)
    
    def forward(self, img):    
        x = self.model(img)
        return x


class DiscriminatorLight(nn.Module):
    """Discriminator Class"""
    def __init__(self, nch_input, nfilters=32, nlayers=3):
        
        super(DiscriminatorLight, self).__init__()
        
        # head
        model = [nn.Conv2d(nch_input, nfilters, kernel_size=4, stride=2, padding=1),
                     nn.LeakyReLU(0.2, True)]
        
        fl = nfilters
        for n in range(1, nlayers):
            model += [nn.Conv2d(fl, fl, kernel_size=4, stride=2, padding=1),
                              nn.InstanceNorm2d(fl),
                              nn.LeakyReLU(0.2, True)]
        
        
        model += [nn.Conv2d(fl, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*model)
    
    def forward(self, img):    
        x = self.model(img)
        return x

class DiscriminatorFeatures(nn.Module):
    def __init__(self, in_features, nf):
        super(DiscriminatorFeatures, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, nf),
                              nn.LeakyReLU(0.2, True),
                              nn.Linear(nf, 1))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)
                