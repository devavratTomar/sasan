import torch.nn as nn
import torch.nn.functional as F
import torch

class GradientLoss(nn.Module):
    def __init__(self, use_gpu=True):
        super(GradientLoss, self).__init__()
        self.l1 = nn.L1Loss()

        self.grad_x = torch.Tensor([[1., 0., -1.],
                                    [2., 0., -2.],
                                    [1., 0., -1.]]).view((1, 1, 3, 3))
        
        self.grad_y = torch.Tensor([[ 1.,   2.,   1.],
                                    [ 0.,   0.,   0.],
                                    [-1.,  -2.,  -1.]]).view((1, 1, 3, 3))

    def image_gradient(self, img):
        self.grad_x = self.grad_x.cuda(img.device)
        self.grad_y = self.grad_y.cuda(img.device)
        return F.conv2d(img, self.grad_x), F.conv2d(img, self.grad_y)

    def forward(self, fake_img, real_img):
        fake_grad_x, fake_grad_y = self.image_gradient(fake_img)
        real_grad_x, real_grad_y = self.image_gradient(real_img)

        loss = self.l1(torch.abs(fake_grad_x), torch.abs(real_grad_x)) + self.l1(torch.abs(fake_grad_y), torch.abs(real_grad_y))

        return loss
