import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY




class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-gaussian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            kernel = torch.Tensor([[math.exp(-(x**2 + y**2)/(2*0.5**2)) / (2*math.pi*0.5**2) for x in range(-3//2, 3//2)]
                           for y in range(-3//2, 3//2)])
            kernel = kernel / torch.sum(kernel)
            kernel = kernel.resize(1, 1, 3, 3)
            for i in range(self.out_planes):
                self.mask[i] = kernel
            # self.mask = kernel
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes): 
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB



class ECB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='prelu', with_idt = False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type
        
        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError(f'The type of {self.act_type} activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv3x3(x)     + \
                self.conv1x1_3x3(x) + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            RK, RB = self.rep_params()
            y = F.conv2d(input=x, weight=RK, bias=RB, stride=1, padding=1) 
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        K0, B0 = self.conv3x3.weight, self.conv3x3.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        RK, RB = (K0+K1+K2+K3+K4), (B0+B1+B2+B3+B4)

        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt
        return RK, RB


@ARCH_REGISTRY.register()
class ECBSR(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(ECBSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        for i in range(self.module_nums):
            backbone += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        backbone += [ECB(self.channel_nums, self.colors*self.scale*self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)]

        self.backbone = nn.Sequential(*backbone)
        self.upsampler = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        # y = self.backbone(x) + x # only work in ycbcr space (1 channel)
        y = self.backbone(x)
        y = self.upsampler(y)
        y = y + F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        return y

@ARCH_REGISTRY.register()
class TaskSyncECBSR(nn.Module):
    def __init__(self, module_nums, channel_nums, with_idt, act_type, scale, colors):
        super(TaskSyncECBSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.with_idt = with_idt
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.denoise_step = self.module_nums // 2
        self.deblur_step = self.module_nums - self.denoise_step

        # 第一步，去噪（小尺度空间）
        self.denoise_inchan_adpt = ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        denoise_module = []
        for i in range(self.denoise_step):
            denoise_module  += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.denoise_module = nn.Sequential(*denoise_module)
        self.denoise_outchan_adpt = ECB(self.channel_nums, self.colors, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)
        
        # 第二步，去模糊（大尺度空间）
        self.deblur_inchan_adpt = ECB(self.colors, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)
        deblur_module = []
        for i in range(self.deblur_step):
            deblur_module += [ECB(self.channel_nums, self.channel_nums, depth_multiplier=2.0, act_type=self.act_type, with_idt = self.with_idt)]
        self.deblur_module = nn.Sequential(*deblur_module)
        self.deblur_outchan_adpt = ECB(self.channel_nums, self.colors* self.scale * self.scale, depth_multiplier=2.0, act_type='linear', with_idt = self.with_idt)
        
        self.upsampler = nn.PixelShuffle(self.scale)


    def forward(self, x):
        denoise_skip = self.denoise_inchan_adpt(x)
        x = self.denoise_module(denoise_skip) + denoise_skip
        x = self.denoise_outchan_adpt(x)
        denoise_out = x

        deblur_skip = self.deblur_inchan_adpt(x)
        x = self.deblur_module(deblur_skip) + deblur_skip
        x = self.deblur_outchan_adpt(x)
        deblur_out = self.upsampler(x)

        if not self.training:
            return deblur_out
        return denoise_out, deblur_out


class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu'):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
        self.act  = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

class PlainSR(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors, export_norm = False):
        super(PlainSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None

        backbone = []
        backbone += [Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)]
        for i in range(self.module_nums):
            backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        backbone += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear')]
        self.backbone = nn.Sequential(*backbone)

        self.upsampler = nn.PixelShuffle(self.scale)
        self.export_norm = export_norm
    
    def forward(self, x):
        # y = self.backbone(x) + x # only work in ycbcr space (1 channel)
        y = self.backbone(x)
        y = self.upsampler(y)
        y = y + F.interpolate(x, scale_factor=self.scale, mode='bilinear')

        if not self.export_norm:
            y = torch.clamp(y * 255, 0, 255)
        return y


class TaskSyncPlainSR(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, colors, mode = 'test',  export_norm = False):
        super(TaskSyncPlainSR, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.fuse_lower = 0.2
        self.fuse_higher = 0.3
        self.export_norm = export_norm
        self.mode = mode
        self.denoise_step = self.module_nums // 2
        self.deblur_step = self.module_nums - self.denoise_step

        self.denoise_inchan_adpt = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        denoise_module = []
        for i in range(self.denoise_step):
            denoise_module += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        self.denoise_module = nn.Sequential(*denoise_module)
        self.denoise_outchan_adpt = Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors, act_type='linear')
        
        self.deblur_inchan_adpt = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type)
        deblur_module = []
        for i in range(self.deblur_step):
            deblur_module += [Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=self.act_type)]
        self.deblur_module = nn.Sequential(*deblur_module)
        self.deblur_outchan_adpt = Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear')

        self.upsampler = nn.PixelShuffle(self.scale)
  
    
    def forward(self, x):
        denoise_skip = self.denoise_inchan_adpt(x)
        x = self.denoise_module(denoise_skip) + denoise_skip
        x = self.denoise_outchan_adpt(x)
        denoise_out = x
        deblur_skip = self.deblur_inchan_adpt(x)
        x = self.deblur_module(deblur_skip) + deblur_skip
        x = self.deblur_outchan_adpt(x)
        deblur_out = self.upsampler(x)

        if self.export_norm:
            if self.mode == 'test':
                return deblur_out, denoise_out
            return deblur_out
        if self.mode == 'test':
            return torch.clamp(deblur_out*255, 0, 255) , torch.clamp(denoise_out*255, 0, 255) 
        return torch.clamp(deblur_out*255, 0, 255) 

def model_ecbsr_rep(model_ecbsr: ECBSR = None, model_plainsr: PlainSR = None) -> None:
    ## copy weights from ecbsr to plainsr
    depth = len(model_ecbsr.backbone)
    for d in range(depth):
        module = model_ecbsr.backbone[d]
        act_type = module.act_type
        RK, RB = module.rep_params()
        model_plainsr.backbone[d].conv3x3.weight.data = RK
        model_plainsr.backbone[d].conv3x3.bias.data = RB

        if act_type == 'relu':     pass
        elif act_type == 'linear': pass
        elif act_type == 'prelu':  model_plainsr.backbone[d].act.weight.data = module.act.weight.data
        else: raise ValueError('invalid type of activation!')

if __name__ == '__main__':
    model = ECBSR(8,32,0,'relu',3,1,True,False)
    model.to('cuda')
    x = torch.randn(3,1,192,256).to('cuda')
    out1 = model(x)
    print(out1.shape)
    