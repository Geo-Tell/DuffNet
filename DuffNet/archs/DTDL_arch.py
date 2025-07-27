import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import math
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows
def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
class Attn(nn.Module):
    def __init__(self,):
        super().__init__()
        self.window_size = 8
        self.attn = WindowAttention(64, window_size=(self.window_size, self.window_size),
                                    num_heads=4)
        self.norm = nn.LayerNorm(64)

    def forward(self, x, rpi_sa, x_size=(16, 16)):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm(x)
        x = x.view(b, h, w, c)
        # partition windows
        x_windows = window_partition(x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        attn_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c
        attn_x = attn_x.view(b, h * w, c)

        x = shortcut + attn_x
        return x
class ViT(nn.Module):
    def __init__(self,):
        super().__init__()
        self.dim = 64
        self.depth = 6
        self.norm = nn.LayerNorm(self.dim)
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                Attn(),
                FeedForward(64, 64*2)
            ]))

    def forward(self, x, rpi):
        for attn, ff in self.layers:
            x = attn(x, rpi)
            x = ff(x) + x
        return self.norm(x)
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
class AEAR(nn.Module):
    def __init__(self, ):
        super(AEAR, self).__init__()
        self.SA = SelfAttention()
        self.low_high_conv = nn.Conv2d(9, 1, 1, 1, 0)
    def L_filter(self, x, m=0.2):
        n = x.shape[-3]
        t1 = torch.tensor([[0.5, -1, 0.5]], dtype=torch.float32)
        t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        t = t1 * m + t2
        kerl = torch.matmul(t.T, t)
        kerl = kerl.view(1, 1, 3, 3).to(x.device)
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
        t1 = torch.tensor([[-0.5, 1, -0.5]], dtype=torch.float32)
        t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32)
        t = t1 * m + t2
        kerl = torch.matmul(t.T, t)
        kerl = kerl.view(1, 1, 3, 3).to(x.device)
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
        return y
    def forward(self, x):
        x = self.frequency_transform1(self.frequency_transform1(x))
        x = self.FAT(x)
        return x
@ARCH_REGISTRY.register()
class DTDL(nn.Module):
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
        super(DTDL, self).__init__()
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if True:
            # for classical SR
            self.conv_first = nn.Conv2d(1, 64, 3, 1, 1)
            self.net = ViT()
            self.up = Upsample(6, 64)
            self.conv = nn.Conv2d(64, 1, 3, 1, 1)
            self.window_size = 8
            # self.aear = AEAR()

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
    def calculate_rpi_sa(self):
        # calculate relative position index for SA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index
    def forward(self, low, high):
        high = self.conv_first(high).view(-1, 64, 16*16).permute(0, 2, 1)
        high = self.net(high, self.calculate_rpi_sa()).permute(0, 2, 1).view(-1, 64, 16, 16)
        x = self.conv(self.up(low + high))
        # x = self.aear(x)
        return x

# model = DTDL()
# low = torch.rand(32, 1, 16, 16)
# high = torch.rand(32, 1, 16, 16)
# x = model(low, high)
# print(x.shape)

