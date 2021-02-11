import torch
import torch.nn as nn
import os
from utilities.util import load_network

from models.discriminator import Discriminator, DiscriminatorLight, DiscriminatorFeatures
from models.generator import GeneratorED
from models.attention import UnetAttention

from loss import GANLoss, OrthonormalityLoss, CrossEntropyLossWeighted
import torch.nn.functional as F
import kornia
from pytorch_msssim import SSIM

import random

from utilities.image_pool import ImagePool

class TrainerCycleGANMSMultiDSeg(object):
    def __init__(self, opt):
        self.opt = opt
        
        # attention models
        self.attention_model_A = UnetAttention(opt.input_ch, 5, opt.n_attention)
        self.attention_model_B = UnetAttention(opt.input_ch, 5, opt.n_attention)


        # load pre-trained attention A?
        self.attention_model_A = load_network(self.attention_model_A, 'Attention_A', 'latest', './checkpoints_attention_mr')

        # freeze the weights
        for param in self.attention_model_A.parameters():
            param.requires_grad = False

        # generators models
        self.G_B = GeneratorED(opt.input_ch, opt.n_attention)
        self.G_A = GeneratorED(opt.input_ch, opt.n_attention)
        
        # adversarial networks
        self.D_B = Discriminator(opt.input_ch, opt.nf, 4)
        self.D_A = Discriminator(opt.input_ch, opt.nf, 4)
        
        # adversarial segmentation
        self.D_segs = self.get_seg_discriminators(opt.nf)
        
        self.D_feat_A = DiscriminatorFeatures(64*64*128, 128)
        self.D_feat_B = DiscriminatorFeatures(64*64*128, 128)
        
        if opt.continue_train:
            self.G_B = load_network(self.G_B, 'G_B', 'latest', opt.checkpoints_dir)
            self.G_A = load_network(self.G_A, 'G_A', 'latest', opt.checkpoints_dir)

            self.D_B = load_network(self.D_B, 'D_B', 'latest', opt.checkpoints_dir)
            self.D_A = load_network(self.D_A, 'D_A', 'latest', opt.checkpoints_dir)
            self.D_feat_A = load_network(self.D_feat_A, 'D_feat_A', 'latest', opt.checkpoints_dir)
            self.D_feat_B = load_network(self.D_feat_B, 'D_feat_B', 'latest', opt.checkpoints_dir)
            
            self.load_seg_discriminators()

            self.attention_model_A = load_network(self.attention_model_A, 'Attention_A', 'latest', opt.checkpoints_dir)
            self.attention_model_B = load_network(self.attention_model_B, 'Attention_B', 'latest', opt.checkpoints_dir)
        
        if len(opt.gpu_ids) > 0:
            self.G_B = self.G_B.cuda()
            self.G_A = self.G_A.cuda()

            self.D_B = self.D_B.cuda()
            self.D_A = self.D_A.cuda()
            self.D_feat_A = self.D_feat_A.cuda()
            self.D_feat_B = self.D_feat_B.cuda()
            
            for i in range(len(self.D_segs)):
                self.D_segs[i] = self.D_segs[i].cuda()
                
            self.attention_model_A = self.attention_model_A.cuda()
            self.attention_model_B = self.attention_model_B.cuda()
        
        # losses
        if opt.gan_type == 'vanilla':
            self.criterian_gan = GANLoss('vanilla')
        elif opt.gan_type == 'lsgan':
            self.criterian_gan = GANLoss('lsgan')
        
        else:
            raise ValueError('Gan type not supported')
        
        self.criterain_l1 = nn.L1Loss()
        self.criterian_ssim = SSIM(1.0, win_size=7, channel=opt.input_ch, spatial_dims=2, nonnegative_ssim=True)
        self.criterian_ssim_attention = SSIM(1.0, win_size=7, channel=opt.n_attention, spatial_dims=2, nonnegative_ssim=True)
        
        self.criterian_ce  = CrossEntropyLossWeighted()
        self.criterian_dice = kornia.losses.DiceLoss()
        self.criterian_ortho = OrthonormalityLoss(opt.n_attention)

        self.optimizer_G, self.optimizer_D = self.create_optimizers()


    def get_seg_discriminators(self, nf):
        d_seg = []
        for i in range(self.opt.n_classes - 1):
            d_seg.append(DiscriminatorLight(1, nf, 3))

        return d_seg


    def load_seg_discriminators(self):
        for i in range(len(self.D_segs)):
            self.D_segs[i] = load_network(self.D_segs[i], 'D_seg' + str(i), 'latest', self.opt.checkpoints_dir)

    def save_seg_discriminators(self, epoch):
        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')
        for i in range(len(self.D_segs)):
            d_seg_name = '%s_net_%s.pth' % (epoch, 'D_seg' + str(i))
            torch.save(self.D_segs[i].state_dict(), os.path.join(saved_model_path, d_seg_name))
            
    def create_optimizers(self):
        G_params = list(self.G_A.parameters()) + list(self.G_B.parameters())
        
        # different learning rate for attention models
        optimizer_G = torch.optim.Adam(G_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=0.0002)
        optimizer_G.add_param_group({
            'params': list(self.attention_model_B.parameters()) + list(self.attention_model_A.parameters()),
            'lr': self.opt.lr * self.opt.mult,
            'mult': self.opt.mult,
            'weight_decay':0.0002
        })
        
        D_params = list(self.D_A.parameters()) + list(self.D_B.parameters()) + list(self.D_feat_A.parameters()) + list(self.D_feat_B.parameters())

        for i in range(len(self.D_segs)):
            D_params += list(self.D_segs[i].parameters())
        optimizer_D = torch.optim.Adam(D_params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999), weight_decay=0.0002)
        
        return optimizer_G, optimizer_D

    def discriminate_imgs(self, D, real_img, fake_img):
        fake_predict_img = D(fake_img)
        real_predict_img = D(real_img)

        return real_predict_img, fake_predict_img

    def discriminate_segs(self, D_segs, seg):
        out = []

        for i in range(len(D_segs)):
            out.append(D_segs[i](seg[:, (i+1):(i+2), ...]))
        
        return out

    def ssim_loss(self, img1, img2):
        return 1.0 - self.criterian_ssim((img1 + 1.0)/(2.0), (img2 + 1.0)/(2.0))

    def seg_loss(self, inputs, targets):
        loss = self.criterian_ce(inputs, targets) + 0.2*self.criterian_dice(inputs, targets)
        return loss

    def one_hot(self, targets):    
        targets_extend=targets.clone()
        targets_extend.unsqueeze_(1) # convert to Nx1xHxW
        one_hot = torch.cuda.FloatTensor(targets_extend.size(0), self.opt.n_classes, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)
        
        return one_hot

    def one_hot_psudo(self, logits):
        return F.softmax(logits, dim=1)
    
    def run_g_step(self, img_A, seg_A, img_B):
        self.optimizer_G.zero_grad()
        
        attentions_A, logits_A = self.attention_model_A(img_A)
        fake_B, en_A = self.G_B(img_A, attentions_A)
        fake_AA, fake_en_B = self.G_A(img_A, attentions_A)

        attentions_fake_B, logits_fake_B = self.attention_model_B(fake_B)

        img_A_r, _ = self.G_A(fake_B, attentions_fake_B)

        attentions_B, logits_B = self.attention_model_B(img_B)
        fake_A, en_B = self.G_A(img_B, attentions_B)
        
        fake_BB, fake_en_A = self.G_B(img_B, attentions_B)

        attentions_fake_A, logits_fake_A = self.attention_model_A(fake_A) # was detached before

        img_B_r, _ = self.G_B(fake_A, attentions_fake_A)

        ## losses
        # adversarial
        loss_G_adver = self.criterian_gan(self.D_B(fake_B), True) + self.criterian_gan(self.D_A(fake_A), True)

        fake_seg_predict = self.discriminate_segs(self.D_segs, F.softmax(logits_fake_A, dim=1))\
                             + self.discriminate_segs(self.D_segs, F.softmax(logits_B, dim=1))
        
        loss_seg_adver = 0.0

        for item in fake_seg_predict:
            loss_seg_adver += self.criterian_gan(item, True)

        loss_seg_adver = loss_seg_adver/5.0

        loss_feat_adver = self.criterian_gan(self.D_feat_B(fake_en_B), True) + self.criterian_gan(self.D_feat_A(fake_en_A), True)

        loss_seg = self.seg_loss(logits_fake_B, seg_A) + self.seg_loss(logits_A, seg_A)

        # use sudo labels to train attention B
        
        loss_seg_psudo = self.criterain_l1(logits_B, logits_fake_A.detach()) + self.seg_loss(logits_B, torch.argmax(logits_fake_A.detach(), dim=1))
        
        loss_atten_psudo = self.criterain_l1(attentions_B, attentions_fake_A) +\
                           self.criterain_l1(attentions_fake_B, attentions_A) # was detach before

        # cycle reconstruction loss
        loss_cycle = self.criterain_l1(img_A_r, img_A) + self.criterain_l1(img_B_r, img_B) + self.ssim_loss(img_A_r, img_A) + self.ssim_loss(img_B_r, img_B)

        # identity loss
        loss_id = self.criterain_l1(fake_AA, img_A) + self.criterain_l1(fake_BB, img_B) + self.ssim_loss(fake_AA, img_A) + self.ssim_loss(fake_BB, img_B)

        # orthogonality of attentions
        loss_ortho = self.criterian_ortho(attentions_A) + self.criterian_ortho(attentions_B)

        loss_G_total = self.opt.lambda_c*loss_cycle + self.opt.lambda_seg_p*loss_seg_psudo + self.opt.lambda_adver*(loss_seg_adver + loss_G_adver + loss_feat_adver) \
                       + self.opt.lambda_id*(loss_id) + self.opt.lambda_seg*loss_seg + self.opt.lambda_orth*loss_ortho + 0.1*loss_atten_psudo

        loss_G_total.backward()
        self.optimizer_G.step()

        self.fake_A = fake_A.detach()
        self.fake_B = fake_B.detach()
        self.seg_A, self.seg_B = torch.argmax(logits_A, dim=1), torch.argmax(logits_B, dim=1)
        self.seg_fake_A = torch.argmax(logits_fake_A, dim=1)

        self.attentions_A = (2*F.softmax(attentions_A.detach(), dim=1) -1).view(-1, 1, attentions_A.size(2), attentions_A.size(3))
        self.attentions_B = (2*F.softmax(attentions_B.detach(), dim=1) -1).view(-1, 1, attentions_B.size(2), attentions_B.size(3))

        # visualization
        self.all_losses_g = {
            'G_adver': loss_G_adver.detach(),
            'Seg_adver': loss_seg_adver.detach(),
            'Seg_loss_fake_B_real_A': loss_seg.detach(),
            'Seg_loss_psudo_B': loss_seg_psudo.detach(),
            'Loss_atten_psudo':loss_atten_psudo.detach(),
            'Loss_cycle': loss_cycle.detach(),
            'Loss_id': loss_id.detach(),
            'Loss_ortho': loss_ortho.detach(),
            'Loos_feat_adver':loss_feat_adver.detach(),
            'all_G':loss_G_total.detach()
        }
    
    def run_d_step(self, img_A, seg_A, img_B):
        self.optimizer_D.zero_grad()
        attentions_B, logits_B = self.attention_model_B(img_B)
        
        fake_A, en_B = self.G_A(img_B, attentions_B)
        _, logits_fake_A = self.attention_model_A(fake_A)

        real_img_predict_A, fake_img_predict_A = self.discriminate_imgs(self.D_A, img_A, fake_A)
        loss_dis_A = self.criterian_gan(real_img_predict_A, True) + self.criterian_gan(fake_img_predict_A, False)

        attentions_A, logits_A = self.attention_model_A(img_A)
        
        fake_B, en_A = self.G_B(img_A, attentions_A)
        _, logits_fake_B = self.attention_model_B(fake_B)
        _, fake_en_B = self.G_A(img_A, attentions_A)

        _,  fake_en_B  = self.G_A(img_A, attentions_A)
        _, fake_en_A   = self.G_B(img_B, attentions_B)

        real_img_predict_B, fake_img_predict_B = self.discriminate_imgs(self.D_B, img_B, fake_B)
        loss_dis_B = self.criterian_gan(real_img_predict_B, True) + self.criterian_gan(fake_img_predict_B, False)


        real_seg_predict = self.discriminate_segs(self.D_segs, F.softmax(logits_A, dim=1))
        
        fake_seg_predict = self.discriminate_segs(self.D_segs, F.softmax(logits_fake_A, dim=1)) \
                             + self.discriminate_segs(self.D_segs, F.softmax(logits_B, dim=1))

        loss_dis_seg = 0.0
        for item in real_seg_predict:
            loss_dis_seg += self.criterian_gan(item, True)

        for item in fake_seg_predict:
            loss_dis_seg += self.criterian_gan(item, False)                        
        
        loss_dis_seg = loss_dis_seg/5.0

        real_feat_A, fake_feat_A = self.discriminate_imgs(self.D_feat_A, en_A, fake_en_A)
        real_feat_B, fake_feat_B = self.discriminate_imgs(self.D_feat_B, en_B, fake_en_B)

        loss_dis_feat = self.criterian_gan(real_feat_A, True) + self.criterian_gan(real_feat_B, True) + self.criterian_gan(fake_feat_A, False) + self.criterian_gan(fake_feat_B, False)

        loss_D_total =  0.5*self.opt.lambda_adver*(loss_dis_A + loss_dis_B + loss_dis_seg + loss_dis_feat)

        loss_D_total.backward()
        self.optimizer_D.step()

        self.all_losses_d = {
            'D_A': loss_dis_A.detach(),
            'D_B': loss_dis_B.detach(),
            'D_seg': loss_dis_seg.detach(),
            'D_feat':loss_dis_feat.detach(),
            'all_D':loss_D_total.detach()
        }
    
    def get_latest_losses(self):
        return {**self.all_losses_g, **self.all_losses_d}

    def update_learning_rate(self, lr):
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.optimizer_G.param_groups:
            mult = param_group.get('mult', 1)
            param_group['lr'] = lr * mult

        print('Updating learning rate to %f' % lr)
    
    def run_test(self, img_A, img_B):
        attentions_A, logits_A = self.attention_model_A(img_A)
        fake_B, _ =   self.G_B(img_A, attentions_A.detach())

        attentions_B, logits_B = self.attention_model_B(img_B)
        fake_A, _ = self.G_A(img_B, attentions_B.detach())

        _, logits_fake_A = self.attention_model_A(fake_A)

        return fake_A.detach(), fake_B.detach(), torch.argmax(logits_A, dim=1), torch.argmax(logits_B, dim=1), torch.argmax(logits_fake_A, dim=1)

    def save(self, epoch):
        g_A_name = '%s_net_%s.pth' % (epoch, 'G_A')
        d_A_name = '%s_net_%s.pth' % (epoch, 'D_A')
        
        g_B_name = '%s_net_%s.pth' % (epoch, 'G_B')
        d_B_name = '%s_net_%s.pth' % (epoch, 'D_B')
        att_A_name = '%s_net_%s.pth' % (epoch, 'Attention_A')
        att_B_name = '%s_net_%s.pth' % (epoch, 'Attention_B')

        d_feat_A_name = '%s_net_%s.pth' % (epoch, 'D_feat_A')
        d_feat_B_name = '%s_net_%s.pth' % (epoch, 'D_feat_B')
        
        saved_model_path = os.path.join(self.opt.checkpoints_dir, 'models')
        
        torch.save(self.G_A.state_dict(), os.path.join(saved_model_path, g_A_name))
        torch.save(self.D_A.state_dict(), os.path.join(saved_model_path, d_A_name))

        torch.save(self.G_B.state_dict(), os.path.join(saved_model_path, g_B_name))
        torch.save(self.D_B.state_dict(), os.path.join(saved_model_path, d_B_name))
        torch.save(self.attention_model_A.state_dict(), os.path.join(saved_model_path, att_A_name))
        torch.save(self.attention_model_B.state_dict(), os.path.join(saved_model_path, att_B_name))
        self.save_seg_discriminators(epoch)

        torch.save(self.D_feat_A.state_dict(), os.path.join(saved_model_path, d_feat_A_name))
        torch.save(self.D_feat_B.state_dict(), os.path.join(saved_model_path, d_feat_B_name))
