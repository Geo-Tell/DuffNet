import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

class BottleneckSR(nn.Module):
    """适用于超分辨率的Bottleneck块"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckSR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        # 超分辨率任务中通常不使用下采样
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out
class ResNetSR(nn.Module):
    """用于超分辨率任务的ResNet backbone"""

    def __init__(self, block, layers, upscale_factor=4):
        super(ResNetSR, self).__init__()
        self.in_channels = 64
        self.upscale_factor = upscale_factor

        # 修改初始卷积层 - 去除大的stride和pooling
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 残差块组 - 去除下采样
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.Conv2d(512 * block.expansion, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor // 2),  # 先上采样2倍
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor // 2),  # 再上采样2倍 (总共4倍)
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 输出RGB图像
        )

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.upsample(x)
        return x
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
class ResNet(nn.Module):
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
        super(ResNet, self).__init__()
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if True:
            # for classical SR
            self.resnet_low = ResNetSR(BottleneckSR, [3, 4, 6, 3], 4)
            self.resnet_high = ResNetSR(BottleneckSR, [3, 4, 6, 3], 4)

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
