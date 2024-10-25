import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, \
    random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F


@MODEL_REGISTRY.register()
class DepthAwareModel(SRGANModel):
    """RealESRGAN Model"""

    def __init__(self, opt):
        super(DepthAwareModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()


    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        # self.lq = self.lq.mean(dim=1, keepdim=True)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            # self.gt = self.gt.mean(dim=1, keepdim=True)
            self.gt_usm = self.usm_sharpener(self.gt)

        if 'add' in data:
            self.add = data['add'].to(self.device)

        if 'dep' in data:
            self.depth = data['dep'].to(self.device)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(DepthAwareModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
    
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq, self.add, self.depth)[0]
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq, self.add, self.depth)[0]
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt
        percep_gt = self.gt
        gan_gt = self.gt

        if self.opt['l1_gt_usm']:
            l1_gt = self.gt_usm
        if self.opt['percep_gt_usm']:
            percep_gt = self.gt_usm
        if self.opt['gan_gt_usm']:
            gan_gt = self.gt_usm

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.add, self.depth)

        if not isinstance(self.output, list):
            self.output = [self.output]

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):

            # content loss
            if self.cri_content:
                l_g_content_final = self.cri_content(self.output[0], l1_gt)
                l_g_content_fine = self.cri_content(self.output[1], l1_gt)
                l_g_total += l_g_content_final
                l_g_total += l_g_content_fine
                loss_dict['l_g_content_final'] = l_g_content_final
                loss_dict['l_g_content_fine'] = l_g_content_fine
            
            # deep edge loss
            if self.cri_edge:
                l_g_edge = self.cri_edge(self.output[0], l1_gt)
                l_g_total += l_g_edge
                loss_dict['l_g_edge'] = l_g_edge
            
            # wavelet Loss
            if self.cri_freq:
                l_g_freq = self.cri_freq(self.output[0], l1_gt)
                l_g_total += l_g_freq
                loss_dict['l_g_freq'] = l_g_freq

            # sobel loss
            if self.cri_sobel:
                l_g_sobel = self.cri_sobel(self.output[0], l1_gt)
                l_g_total += l_g_sobel
                loss_dict['l_g_sobel'] = l_g_sobel

            if self.cri_laplacian:
                l_g_laplacian = self.cri_laplacian(self.output[0], l1_gt)
                l_g_total += l_g_laplacian

            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output[1], l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix
            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output[0], percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style
            # gan loss
            if self.cri_gan:
                fake_g_pred = self.net_d(self.output[0])
                l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                l_g_total += l_g_gan
                loss_dict['l_g_gan'] = l_g_gan


            loss_dict['l_g_total'] = l_g_total
            l_g_total.backward()
            self.optimizer_g.step()


        if self.cri_gan:
            # optimize net_d
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(gan_gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output[0].detach().clone())  # clone for pt1.9
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()
            self.optimizer_d.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
