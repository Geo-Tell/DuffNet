import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from math import sqrt
import math


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.up = Upsample(4, 64)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input(x)
        residual = x
        out = self.relu(x)
        out = self.residual_layer(out)
        out = torch.add(out, residual)
        out = self.output(self.up(out))
        return out
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        elif scale == 6:
            m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class SelfAttention(nn.Module):
    def __init__(self, emb_size=256):
        super(SelfAttention, self).__init__()
        self.emb_size = emb_size
        self.key = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.query = nn.Linear(self.emb_size, self.emb_size, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, self.emb_size)
        queries = self.query(x)
        keys = self.key(x)

        queries = queries.view(b, -1, 1, self.emb_size)
        keys = keys.view(b, -1, 1, self.emb_size)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = F.softmax(energy, dim=-1)
        # output = attention[:, :, 0:1, :].reshape(b, c, 1, 1)
        return attention
@ARCH_REGISTRY.register()
class VDSR(nn.Module):
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):
        super(VDSR, self).__init__()
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if True:
            # for classical SR
            self.resnet_low = Net()
            self.resnet_high = Net()

            self.low_high_conv = nn.Conv2d(15, 1, 1, 1, 0)
            self.SA = SelfAttention()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def L_filter(self, x, m=0.2):
        n = x.shape[-3]
        t1 = torch.tensor([[0.5, -1, 0.5]], dtype=torch.float32).cuda()
        t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32).cuda()
        t = t1 * m + t2
        kerl = torch.matmul(t.T, t)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        for i in range(n):
            x_out = F.conv2d(input=x[:, i:i + 1, :, :], weight=kerl, bias=None, stride=1, padding=0, dilation=1,
                             groups=1)
            if i == 0:
                x_hy = x_out
            else:
                x_hy = torch.cat([x_hy, x_out], dim=1)
        return x_hy
    def H_filter(self, x, m=0.2):
        n = x.shape[-3]
        t1 = torch.tensor([[-0.5, 1, -0.5]], dtype=torch.float32).cuda()
        t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32).cuda()
        t = t1 * m + t2
        kerl = torch.matmul(t.T, t)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        for i in range(n):
            x_out = F.conv2d(input=x[:, i:i + 1, :, :], weight=kerl, bias=None, stride=1, padding=0, dilation=1,
                             groups=1)
            if i == 0:
                x_hy = x_out
            else:
                x_hy = torch.cat([x_hy, x_out], dim=1)
        return x_hy
    def frequency_transform1(self, x):
        x1 = self.L_filter(x, m=0.05)
        x2 = self.H_filter(x, m=0.05)
        x_hy = torch.cat([x, x1, x2], dim=1)
        return x_hy
    def frequency_transform2(self, x):
        x1 = self.L_filter(x, m=0.05)
        x2 = self.H_filter(x, m=0.05)
        x3 = self.L_filter(x, m=0.1)
        x4 = self.H_filter(x, m=0.1)
        x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
        return x_hy
    def FAT(self, x):
        b, c, h, w = x.shape
        x_sub = x
        A = self.SA(x_sub[:, :, ::4, ::4])
        weight = A.reshape(b, c ** 2, 1, 1)
        x = torch.cat([x] * c, dim=1)
        y = (x * weight).reshape(b, c, c, h, w)
        y = torch.sum(y, dim=2)
        y = self.low_high_conv(y)
        # y = torch.sum(y.reshape(b, 1, c, h, w), dim=2) / c
        return y
    def forward(self, x):
        if True:
            x_low = self.resnet_low(x)
            x_high = self.resnet_high(x)

            x = x_low + x_high

            x = self.frequency_transform2(self.frequency_transform1(x))
            x = self.FAT(x)

        return (x_low, x_high), x

# model = Net()
# x = torch.rand(32, 1, 16, 16)
# y = model(x)
# print(y.shape)

