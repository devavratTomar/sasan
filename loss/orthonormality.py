import torch.nn as nn
import torch

class OrthonormalityLoss(nn.Module):
    def __init__(self, size):
        super(OrthonormalityLoss, self).__init__()
        self.size = size
        self.lower_tr = torch.zeros(size, size).cuda()
        indices = torch.tril_indices(size, size, offset=-1)

        # # lower triangular matrix as mask
        # self.lower_tr[indices[0], indices[1]] = 1.0
        self.id = torch.eye(self.size).cuda()

    def forward(self, attentions):
        x = attentions.view(attentions.shape[0], attentions.shape[1], -1)

        # normalize xx
        x = nn.functional.normalize(x, dim=2)

        #batch_matrix = torch.matmul(x, torch.transpose(x, 1, 2))*self.lower_tr
        batch_matrix = (torch.matmul(x, torch.transpose(x, 1, 2)) - self.id)
        
        
        cost = 2*(batch_matrix**2).sum()/(self.size * (self.size - 1.0))

        return cost
        
