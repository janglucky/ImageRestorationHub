import torch
import torch.nn.functional as F
from einops import rearrange
import torch.nn as nn
from basicsr.archs.arch_util import LayerNorm2d

class ClarityEstimator(nn.Module):
    def __init__(self, in_chan, patch_size, scale_level=4, module_num=8,  module_chan=64, middle_feat=15, num_class=3, num_experts=5, use_bias=True):
        super(ClarityEstimator, self).__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_chan, module_chan, kernel_size=3, stride=1,padding=1, bias=use_bias),
            nn.ReLU()
        )
        backbone = []
        for _ in range(module_num):
            backbone.append(nn.Conv2d(module_num, module_num, kernel_size=3, stride=1, padding=1, bias=use_bias))
            backbone.append(nn.ReLU())

        self.backbone = nn.Sequential(*backbone)

        clasifiers = []
        weight_heads = []
        for i in range(scale_level):
            clasifiers.append(nn.Sequential(
                nn.AvgPool2d(kernel_size=patch_size * (i+1)),
                nn.Conv2d(module_chan, middle_feat, kernel_size=1,bias=use_bias),
                nn.Conv2d(middle_feat, num_class, kernel_size=1,bias=use_bias)
            ))

            weight_heads.append(nn.Sequential(
                nn.Conv2d(num_class, middle_feat, kernel_size=1,bias=use_bias),
                nn.Conv2d(middle_feat, num_experts, kernel_size=1,bias=use_bias)
            ))

        self.clasifiers = clasifiers
        self.weight_heads = weight_heads
        self.scale_levle = scale_level

    def forward(self, x):
        x = self.conv_in(x)
        x = self.backbone(x)

        clas = []
        vs = []
        for clasifier, weight in zip(self.clasifiers, self.weight_heads):
            x = clasifier(x)
        return x, v


class ExpertConv2d(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride=1, padding=1, dilation=1, patch_size=64, bias=True, K = 5):
        super(ExpertConv2d, self).__init__()

        self.K = K
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.patch_size = patch_size
        self.weight = nn.Parameter(torch.randn(K, out_chan, in_chan // self.groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_chan), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x, v):
        """
        x 输入特征图
        v 加权
        """
        y = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1=self.patch_size, p2=self.patch_size)
        v = rearrange(v, 'b k h w -> (b h w) k')

        batch_size, in_planes, height, width = y.size()
        y = y.contiguous().view(1, -1, height, width) # 1, b * in_chan, height, width
        weight = self.weight.view(self.K, -1) # K, n

        aggregate_weight = torch.mm(v, weight).view(-1, in_planes, self.kernel_size , self.kernel_size) # b * out_chan, in_chan, k, k

        if self.bias is not None:
            aggregate_bias = torch.mm(v, self.bias).view(-1)
            output = F.conv2d(y, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        else:
            output = F.conv2d(y, weight=aggregate_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_chan, output.size(-2), output.size(-1))
        output = rearrange(output, '(b h w) c p1 p2 -> b c (h p1) (w p2)',h=512//self.patch_size, w = 512//self.patch_size)
        return output
    

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class CMOEBlock(nn.Module):
    def __init__(self, c, patch_size, K=5, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = ExpertConv2d(in_chan=c, out_chan=dw_channel, patch_size=patch_size, kernel_size=1, padding=0, stride=1, bias=True, K=K)
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = ExpertConv2d(in_chan=dw_channel, out_chan=dw_channel, patch_size=patch_size, kernel_size=3,padding=1,stride=1,bias=True, K=K)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
        #                        bias=True)
        self.conv3 = ExpertConv2d(in_chan=dw_channel//2, out_chan=c,patch_size=patch_size,kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.sca = ExpertConv2d(in_chan=dw_channel//2, out_chan=dw_channel//2, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)


        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = ExpertConv2d(in_chan=c,out_chan=ffn_channel, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.conv5 = ExpertConv2d(in_chan=ffn_channel//2, out_chan=c, patch_size=patch_size, kernel_size=1,padding=0,stride=1,bias=True, K=K)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inps):
        inp, v = inps
        x = inp

        x = self.norm1(x)

        x = self.conv1(x, v)
        x = self.conv2(x, v)
        keep_x = self.sg(x)
        
        x = self.avg(keep_x)
        x = x * self.sca(keep_x, v)
        x = self.conv3(x, v)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y), v)
        x = self.sg(x)
        x = self.conv5(x, v)

        x = self.dropout2(x)

        return y + x * self.gamma
    
class PCAMoENet(nn.Module):

    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], patch_size = 32, K =5):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CMOEBlock(chan, patch_size=patch_size,K=K) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[CMOEBlock(chan, patch_size=patch_size,K=K) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CMOEBlock(chan, patch_size=patch_size,K=K) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, v):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            print(x.shape, v.shape)
            x = encoder([x, v])
            encs.append(x)
            x = down(x)

        x = self.middle_blks([x, v])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder([x, v])

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
    

if __name__ == '__main__':
    patch_size = 32
    ce_model = ClarityEstimator(3, patch_size, 4, 8,32,15,3,5,True)
    model = PCAMoENet(3,width=64,enc_blk_nums=[1,1,1,28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1], patch_size=32, K=5)


    # width: 64
    # enc_blk_nums: [1, 1, 1, 28]
    # middle_blk_num: 1
    # dec_blk_nums: [1, 1, 1, 1]
    # dep_blk_chans: [6, 64, 128, 256, 128, 64, 1]

    x = torch.randn(2,3,512, 512)
    c, v = ce_model(x)
    print(x.shape, v.shape)
    out = model(x, v)
    print(out.shape)

