import torch.nn as nn
import torch

from torch.nn import functional as F
from torch.autograd import grad

class WganGpLoss(nn.Module):
    def __init__(self, penalty=10.):
        super().__init__()
        self.penalty = penalty

    def forward(self, prediction, is_real):
        """
        Computes the wgan loss without grad penlatiy. Should call cal_grad_penalty to compute gradient penalty term.
        """
        loss = prediction.mean()

        if is_real:
            return -loss
        else:
            return loss

    def cal_gradient_penalty(self, img, atten_img, fake_img, atten_fake_img, D):
        b_size = img.size(0)
        eps = torch.rand(b_size, 1, 1, 1).cuda(img.device)
        x_hat = eps * img.data + (1 - eps) * fake_img.data
        x_hat.requires_grad = True
        
        hat_predict = D(x_hat)

        # take gradient and flatten
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0].view(b_size, -1)

        grad_penalty = (( grad_x_hat.norm(2, dim=1) - 1.0) ** 2).mean()
        return grad_penalty
            
class WganGPR1(nn.Module):
    def __init__(self, penalty=5.):
        super().__init__()
        self.penalty = penalty
        
    def forward(self, fake_predict, mode, real_scores=None, grad_real=None):
        
        if mode == 'discriminator':
            real_predict = F.softplus(-real_scores).mean()

            grad_penalty = self.penalty*(
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()

            fake_predict = F.softplus(fake_predict).mean()

            loss = fake_predict + real_predict + grad_penalty
            return loss
        
        elif mode == 'generator':
            loss = F.softplus(-fake_predict).mean()
            return loss