import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualSparseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=32, num_grow_ch=32, act_type = 'relu'):
        super(ResidualSparseBlock, self).__init__()
        self.conv_in = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)

        if act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        # initialization
        default_init_weights([self.conv_in, self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x0 = self.act(self.conv_in(x))
        x1 = self.act(self.conv1(x0))
        x2 = self.act(self.conv2(x1))
        x3 = self.act(self.conv3(torch.cat((x1, x2), 1)))
        return x3 + x0

# @ARCH_REGISTRY.register()
class SparseRRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, act_type = 'relu', phase = 'train'):
        super(SparseRRDBNet, self).__init__()
        self.scale = scale
        self.phase = phase
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_denoise = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1)

        self.conv_srin = nn.Conv2d(num_in_ch, num_feat, 3,1, 1)
        self.body = make_layer(ResidualSparseBlock, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, act_type=act_type)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_fit = nn.Conv2d(num_feat, num_feat * scale* scale, 3, 1, 1)
        self.upsampler = nn.PixelShuffle(upscale_factor=scale)

        # final output
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)


        if act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            raise ValueError(f'The type of activation {act_type} if not support!')

    def forward(self, x):
        feat = self.act(self.conv_first(x))
        denoise = self.act(self.conv_denoise(feat))
        feat = self.act(self.conv_srin(denoise))
        body_feat = self.act(self.conv_body(self.body(feat)))
        feat = feat + body_feat

        # upsample
        feat = self.upsampler(self.act(self.conv_fit(feat)))
        out = self.conv_last(feat)
        if self.phase in ['test', 'export']:
            out = torch.clamp(out * 255, 0, 255)
        return [out, denoise]

if __name__ == '__main__':
    model = SparseRRDBNet(1,1,2,32,3,32, 'relu','test')
    x = torch.randn(1,1,512,640)
    torch.onnx.export(
            model,
            x,
            'spareserrdb.onnx',
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
        )