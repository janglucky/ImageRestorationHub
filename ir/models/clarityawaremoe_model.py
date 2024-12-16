import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, \
    random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.models.srgan_model import SRGANModel, SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from collections import OrderedDict
from torch.nn import functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger


@MODEL_REGISTRY.register()
class ClarityAwareMoeModel(SRModel):
    """RealESRGAN Model"""

    def __init__(self, opt):
        super(ClarityAwareMoeModel, self).__init__(opt)
        self.usm_sharpener = USMSharp().cuda()

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            for p in self.net_g_ema.parameters():
                p.requires_grad = False
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
                
            self.net_g_ema.eval()


        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        load_key = self.opt['path'].get('param_key_g', None)
        if load_path is not None:
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), load_key)

        # 加载图像清晰度分类网络
        self.net_c = build_network(self.opt['network_c'])
        self.net_c = self.model_to_device(self.net_c)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_c', None)
        load_key = self.opt['path'].get('param_key_c', None)
        if load_path is not None:
            self.load_network(self.net_c, load_path, self.opt['path'].get('strict_load_c', True), load_key)

        self.net_g.train()
        self.net_c.train()

        # define losses
        if train_opt.get('clarity_opt') and train_opt['clarity_opt']['loss_weight'] > 0:
            self.cri_clarity = build_loss(train_opt['clarity_opt']).to(self.device)
        else:
            self.cri_clarity = None
        if train_opt.get('pixel_opt') and train_opt['pixel_opt']['loss_weight'] > 0:
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('content_opt') and train_opt['content_opt']['loss_weight'] > 0:
            self.cri_content = build_loss(train_opt['content_opt']).to(self.device)
        else:
            self.cri_content = None

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, filter(lambda p: p.requires_grad, self.net_g.parameters()), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer c
        optim_type = train_opt['optim_c'].pop('type')
        self.optimizer_c = self.get_optimizer(optim_type, self.net_c.parameters(), **train_opt['optim_c'])
        self.optimizers.append(self.optimizer_c)

    @torch.no_grad()
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)

        self.cond = data['cond'].to(self.device)
        # self.lq = self.lq.mean(dim=1, keepdim=True)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            # self.gt = self.gt.mean(dim=1, keepdim=True)
            self.gt_usm = self.usm_sharpener(self.gt)
        
        if 'labels' in data:
            self.labels = []
            for label in data['labels']:
                self.labels.append(label.to(self.device))


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        super(ClarityAwareMoeModel, self).nondist_validation(
            dataloader, current_iter, tb_logger, save_img)
        self.is_train = True
    
    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                cs, weight = self.net_c(self.lq)
                self.output = self.net_g_ema(self.lq, self.cond, weight)[0]
        else:
            self.net_g.eval()
            with torch.no_grad():
                cs, weight = self.net_c(self.lq, weight)
                self.output = self.net_g(self.lq, self.cond, weight)[0]
            self.net_g.train()

    def optimize_parameters(self, current_iter):
        l1_gt = self.gt

        if self.opt['l1_gt_usm']:
            l1_gt = self.gt_usm

        # optimize net_g
        for p in self.net_c.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.clas, self.weight = self.net_c(self.lq)

        self.output = self.net_g(self.lq, self.cond, self.weight)
        
        if not isinstance(self.output, list):
            self.output = [self.output]

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):

            # clarity loss
            if self.cri_clarity:

                for i, (out, label) in enumerate(zip(self.clas, self.labels)):
                    l_g_clarity = self.cri_clarity(out, label)
                    l_g_total += l_g_clarity
                    loss_dict[f'l_g_clarity{i}'] = l_g_clarity
            # content loss
            if self.cri_content:
                l_g_content = self.cri_content(self.output[0], l1_gt)
                l_g_total += l_g_content
                loss_dict['l_g_content'] = l_g_content
            

            # pixel loss
            if self.cri_pix:
                l_g_pix = self.cri_pix(self.output[1], l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix


            loss_dict['l_g_total'] = l_g_total
            l_g_total.backward()
            self.optimizer_g.step()


        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)
    
    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)

        self.save_network(self.net_c, 'net_c', current_iter)
        self.save_training_state(epoch, current_iter)
