from nnunet.network_architecture.generic_UNet import Generic_UNet
from nnunet.network_architecture.initialization import InitWeights_He
from utilities.util import load_network
from models.segmentation import SegmentationModel

import kornia
import torch.nn as nn
import torch
import os

from loss import CrossEntropyLossWeighted


class SegmentationTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        
        if opt.segmenter=='UNET++':
            norm_op_kwargs = {'eps': 1e-5, 'affine': True}
            dropout_op_kwargs = {'p': 0, 'inplace': True}
            net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
            self.net = Generic_UNet(opt.input_ch, 64, opt.n_classes,
                                    4,
                                    num_conv_per_stage=2,
                                    norm_op=nn.InstanceNorm2d, 
                                    norm_op_kwargs=norm_op_kwargs,
                                    dropout_op_kwargs=dropout_op_kwargs,
                                    nonlin_kwargs=net_nonlin_kwargs,
                                    final_nonlin=lambda x: x,
                                    convolutional_pooling=True, 
                                    convolutional_upsampling=True)
        else:
            print("HERE")
            self.net = SegmentationModel(opt.input_ch, opt.n_classes)

        if opt.continue_train:
            self.net = load_network(self.net, opt.modality, 'latest', opt.checkpoints_dir)
        
        if len(opt.gpu_ids) > 0:
            self.net = self.net.cuda()
        
        # losses
        self.criterian_dice = kornia.losses.DiceLoss()
        self.criterian_bce = CrossEntropyLossWeighted()

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def run_train_step(self, x, seg):
        # clear the gradients
        self.optimizer.zero_grad()

        predictions = self.net(x)
        if self.opt.segmenter == "UNET++":
            predictions = predictions[0]
        self.predictions_index = torch.argmax(predictions, dim=1, keepdim=True)

        # get losses
        dice = self.criterian_dice(predictions, seg)
        cross_entropy = self.opt.lambda_ce*self.criterian_bce(predictions, seg)
        self.losses = {'cross_entropy': cross_entropy, 'dice': dice}
        
        # gradient step
        loss = 0.2*dice + cross_entropy
        loss.backward()
        self.optimizer.step()

    def run_test(self, x):
        # go in eval mode
        self.net.eval()
        predictions = self.net(x)
        if self.opt.segmenter == "UNET++":
            predictions = predictions[0]

        # back to train mode
        self.net.train()

        predictions_index = torch.argmax(predictions.detach(), dim=1, keepdim=True)
        return predictions_index

    def update_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, epoch):
        model_name = '%s_net_%s.pth' % (epoch, self.opt.modality)
        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')

        torch.save(self.net.state_dict(), os.path.join(saved_model_path, model_name))
                    

        
