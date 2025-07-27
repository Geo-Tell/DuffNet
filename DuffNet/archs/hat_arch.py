import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_2tuple, trunc_normal_

from einops import rearrange
class SpectralGatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(64, 33, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        # print('weight',weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(64, 64), dim=(2, 3), norm='ortho')
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):

    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB, self).__init__()

        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
            )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Mlp_shallow(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.drop(x)
        return x

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


class HAB(nn.Module):
    r""" Hybrid Attention Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, 'shift_size must in 0-window_size'

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        # assert seq_len == h * w, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv_X
        conv_x = self.conv_block(x.permute(0, 3, 1, 2))
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = attn_mask
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        attn_x = attn_x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class Patch_Shuffle(nn.Module):
    def __init__(self, s):
        super().__init__()
        self.s = s
        self.padding_number = s % 3
        self.padding = nn.ZeroPad2d(self.padding_number)
        self.patch_number = int((s + 2 * (s % 3)) / 3)
        self.mlp_list = [Mlp(in_features=9).cuda() for _ in range(self.patch_number**2)]
    def split_patch(self, x):

        patch_list = [0 for _ in range(self.patch_number**2)]
        for i in range(self.patch_number):
            for j in range(self.patch_number):
                patch_list[i * self.patch_number + j] = x[:, :, 3*i:3*i+3, 3*j:3*j+3]
        return patch_list
    def tensor_splicing(self, x):
        b, c, _, _ = x[0].shape
        x_shuffle = torch.zeros(b, c, 3*self.patch_number, 3*self.patch_number).cuda()
        for i in range(self.patch_number**2):
            x_shuffle[:, :, 3*(i//self.patch_number):3*(i//self.patch_number)+3,
            3*(i%self.patch_number):3*(i%self.patch_number)+3] = x[i]
        return x_shuffle
    def forward(self, x):
        b, _, c = x.shape
        x = x.view(b, self.s, self.s, c).permute(0, 3, 1, 2)
        patch_list = self.split_patch(self.padding(x))
        patch_shuffle_list = [0 for _ in range(self.patch_number**2)]
        for i in range(self.patch_number**2):
            patch_sample = patch_list[i].contiguous().view(b, c, 9)
            patch_shuffle_sample = self.mlp_list[i](patch_sample)
            patch_shuffle_list[i] = patch_shuffle_sample.contiguous().view(b, c, 3, 3)
        x_shuffle = self.tensor_splicing(patch_shuffle_list)[:, :,
                    self.padding_number:self.padding_number+self.s,
                    self.padding_number:self.padding_number+self.s]
        output = x_shuffle.permute(0, 2, 3, 1).contiguous().view(b, self.s**2, c)
        return output

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x


class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x


class AttenBlocks(nn.Module):
    """ A series of attention blocks for one RHAG.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 upscale=4):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            HAB(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # little_patch_shuffle
        # self.patch_shuffle = Patch_Shuffle(s=int(64 / upscale)).cuda()

        # OCAB
        self.overlap_attn = OCAB(
                            dim=dim,
                            input_resolution=input_resolution,
                            window_size=window_size,
                            overlap_ratio=overlap_ratio,
                            num_heads=num_heads,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            mlp_ratio=mlp_ratio,
                            norm_layer=norm_layer
                            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])
        # x = self.patch_shuffle(x)
        # x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 upscale=4):
        super(RHAG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = AttenBlocks(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            upscale=upscale)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x


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
class HAT(nn.Module):
    r""" Hybrid Attention Transformer
        A PyTorch implementation of : `Activating More Pixels in Image Super-Resolution Transformer`.
        Some codes are based on SwinIR.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

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
        super(HAT, self).__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        self.conv_first_shallow = nn.Conv2d(num_in_ch, 24, 5, 1, 2)
        self.mlp_first_shallow = Mlp_shallow(in_features=24, hidden_features=36, out_features=16)
        self.mlp_high = Mlp(in_features=6, hidden_features=12, out_features=6)
        self.conv_high = nn.Conv2d(6, 1, 3, 1, 1)
        self.upsample_shallow = nn.PixelShuffle(4)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # n = 64
        # # a, b, c = 8/28, 4/28, 1/28
        # a, b, c = 0.8, 0.03, 0.02
        # # a, b, c = 8 / 20, 2 / 20, 1 / 20
        # x = torch.zeros(n ** 2, (n + 2) ** 2).cuda()
        # y = torch.zeros((n + 2) ** 2, n ** 2).cuda()
        # for i in range(n):
        #     for j in range(n):
        #         x[i * n + j, i * (n + 2) + j] = c
        #         x[i * n + j, i * (n + 2) + j + 1] = b
        #         x[i * n + j, i * (n + 2) + j + 2] = c
        #
        #         x[i * n + j, i * (n + 2) + j + n + 2] = b
        #         x[i * n + j, i * (n + 2) + j + n + 3] = a
        #         x[i * n + j, i * (n + 2) + j + n + 4] = b
        #
        #         x[i * n + j, i * (n + 2) + j + 2 * n + 4] = c
        #         x[i * n + j, i * (n + 2) + j + 2 * n + 5] = b
        #         x[i * n + j, i * (n + 2) + j + 2 * n + 6] = c
        # for i in range(n + 2):
        #     for j in range(n + 2):
        #         if i == 0:
        #             if j == 0:
        #                 y[i * (n + 2) + j, (i + 1) * n + 1] = 1
        #             elif j == n + 1:
        #                 y[i * (n + 2) + j, (i + 1) * n + n - 2] = 1
        #             else:
        #                 y[i * (n + 2) + j, (i + 1) * n + j - 1] = 1
        #         elif i == n + 1:
        #             if j == 0:
        #                 y[i * (n + 2) + j, (i - 3) * n + 1] = 1
        #             elif j == n + 1:
        #                 y[i * (n + 2) + j, (i - 3) * n + n - 2] = 1
        #             else:
        #                 y[i * (n + 2) + j, (i - 3) * n + j - 1] = 1
        #         else:
        #             if j == 0:
        #                 y[i * (n + 2) + j, (i - 1) * n + 1] = 1
        #             elif j == n + 1:
        #                 y[i * (n + 2) + j, (i - 1) * n + n - 2] = 1
        #             else:
        #                 y[i * (n + 2) + j, (i - 1) * n + j - 1] = 1
        # self.inv = torch.inverse(torch.matmul(x, y))

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Hybrid Attention Groups (RHAG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                upscale=self.upscale)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)
        self.layers_shared = nn.ModuleList()
        for i_layer in range(2):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=0,  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                upscale=self.upscale)
            self.layers_shared.append(layer)
            self.layers_low = nn.ModuleList()
            for i_layer in range(2):
                layer = RHAG(
                    dim=embed_dim,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    overlap_ratio=overlap_ratio,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=0,  # no impact on SR results
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint,
                    img_size=img_size,
                    patch_size=patch_size,
                    resi_connection=resi_connection,
                    upscale=self.upscale)
                self.layers_low.append(layer)
                self.layers_high = nn.ModuleList()
                for i_layer in range(2):
                    layer = RHAG(
                        dim=embed_dim,
                        input_resolution=(patches_resolution[0], patches_resolution[1]),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        compress_ratio=compress_ratio,
                        squeeze_factor=squeeze_factor,
                        conv_scale=conv_scale,
                        overlap_ratio=overlap_ratio,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=0,  # no impact on SR results
                        norm_layer=norm_layer,
                        downsample=None,
                        use_checkpoint=use_checkpoint,
                        img_size=img_size,
                        patch_size=patch_size,
                        resi_connection=resi_connection,
                        upscale=self.upscale)
                    self.layers_high.append(layer)
                    self.layers_high1 = nn.ModuleList()
                    for i_layer in range(1):
                        layer = RHAG(
                            dim=embed_dim,
                            input_resolution=(patches_resolution[0], patches_resolution[1]),
                            depth=depths[i_layer],
                            num_heads=num_heads[i_layer],
                            window_size=window_size,
                            compress_ratio=compress_ratio,
                            squeeze_factor=squeeze_factor,
                            conv_scale=conv_scale,
                            overlap_ratio=overlap_ratio,
                            mlp_ratio=self.mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=0,  # no impact on SR results
                            norm_layer=norm_layer,
                            downsample=None,
                            use_checkpoint=use_checkpoint,
                            img_size=img_size,
                            patch_size=patch_size,
                            resi_connection=resi_connection,
                            upscale=self.upscale)
                        self.layers_high1.append(layer)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_low = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_high = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_after_body_high1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()
            self.conv_after_body_low = nn.Identity()
            self.conv_after_body_high = nn.Identity()
            self.conv_after_body_high1 = nn.Identity()
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_before_upsample_low = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_before_upsample_high = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_before_upsample_high1 = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.upsample_low = Upsample(upscale, num_feat)
            self.upsample_high = Upsample(upscale, num_feat)
            self.upsample_high1 = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_last1 = nn.Conv2d(1, 1, 3, 1, 1)
            self.conv_last_low = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_last_high = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.conv_last_high1 = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            # self.low_conv = nn.Conv2d(6, 1, 1, 1, 0)
            # self.high_conv = nn.Conv2d(6, 1, 1, 1, 0)
            self.low_high_conv = nn.Conv2d(9, 1, 1, 1, 0)
            # self.fc_last = nn.Linear(64*64, 64*64)
            # self.fft_low = SpectralGatingNetwork()
            # self.fft_high = SpectralGatingNetwork()
            # self.fft1 = SpectralGatingNetwork()
            # self.fft2 = SpectralGatingNetwork()
            # self.fft3 = SpectralGatingNetwork()
            # self.fft4 = SpectralGatingNetwork()
            # self.fft5 = SpectralGatingNetwork()
            # self.fft6 = SpectralGatingNetwork()
            # self.low_high_conv = nn.Conv2d(7, 1, 1, 1, 0)
            # self.low_high_conv_2 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.low_high_conv1 = nn.Conv2d(7, 1, 1, 1, 0)
            # self.low_high_conv1_2 = nn.Conv2d(7, 1, 1, 1, 0)
            # self.low_high_conv2 = nn.Conv2d(2, 1, 1, 1, 0)
            # self.w_low = nn.Parameter(torch.tensor([0.2555, 0.549, 0.973, 0.2555, 0.549, 0.973]))
            # self.w_high = nn.Parameter(torch.tensor([0.2555, 0.549, 0.973, 0.2555, 0.549, 0.973]))
            self.tanh = torch.nn.Tanh()
            # self.weight = nn.Parameter(torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5]))
            self.weight = nn.Parameter(torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))
            # self.w_low = nn.Parameter(torch.tensor([0.2555, 0.549, 0.2555, 0.549]))
            # self.w_list_low = [nn.Parameter(torch.tensor([0.2555, 0.2555])) for i in range(5)]
            # self.w_high = nn.Parameter(torch.tensor([0.2555, 0.549, 0.2555, 0.549]))
            # self.w_list_high = [nn.Parameter(torch.tensor([0.2555, 0.2555])) for i in range(5)]
            # self.w_last = nn.Parameter(torch.tensor([0.2555, 0.549, 0.973, 0.2555, 0.549, 0.973]))
            # self.w_Multi1 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.w_Multi2 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.w_Multi3 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.w_Multi4 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.w_Multi5 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.w_Multi6 = nn.Parameter(torch.tensor([0.2555, 0.2555]))
            # self.low_high_conv2_2 = nn.Conv2d(7, 1, 1, 1, 0)
            # self.patch_convL1 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convL2 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convL3 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convL4 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convH1 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convH2 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convH3 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.patch_convH4 = nn.Conv2d(11, 1, 1, 1, 0)
            # self.low_Conv= nn.Conv2d(15, 1, 1, 1, 0)
            # self.high_Conv = nn.Conv2d(15, 1, 1, 1, 0)
            # self.Multi1 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi2 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi3 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi4 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi5 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi6 = nn.Conv2d(3, 1, 1, 1, 0)
            # self.Multi7 = nn.Conv2d(3, 1, 1, 1, 0)
            self.SA_low = SelfAttention()
            self.SA_high = SelfAttention()
            self.SA = SelfAttention()
            self.SA_gt = SelfAttention()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}
    def Shared_layers(self, x):
        x_size = (x.shape[2], x.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_shared:
            x = layer(x, x_size, params)
        return x
    def forward_features_low(self, x, x_shallow):
        x_size = (x_shallow.shape[2], x_shallow.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}
        for layer in self.layers_low:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x
    def forward_features_high(self, x, x_shallow):
        x_size = (x_shallow.shape[2], x_shallow.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}
        for layer in self.layers_high:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward_features_high1(self, x, x_shallow):
        x_size = (x_shallow.shape[2], x_shallow.shape[3])
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA,
                  'rpi_oca': self.relative_position_index_OCA}
        for layer in self.layers_high1:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def shallow_feature(self, x):
        b, _, h, w = x.shape
        x_ori = x
        x = self.conv_first_shallow(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h*w, 24)
        x = self.mlp_first_shallow(x)
        x = x.view(b, h, w, 16).permute(0, 3, 1, 2)
        x = x_ori + x
        x = self.upsample_shallow(x)
        return x
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference.
        # The original code is very time-consuming for large window size.
        attn_mask = self.calculate_mask(x_size).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def mlp_last(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, h*w)
        x = self.fc_last(x)
        return x.view(b, c, h, w)

    # def L_filter(self, x, m=0.2):
    #     t1 = torch.tensor([[0.5, -1, 0.5]], dtype=torch.float32).cuda()
    #     t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32).cuda()
    #     t = t1 * m + t2
    #     kerl = torch.matmul(t.T, t)
    #     # kerl = torch.tensor(
    #     #     [[0.25 * m ** 2, 0.5 * m * (1 - m), 0.25 * m ** 2], [0.5 * m * (1 - m), (1 - m) ** 2, 0.5 * m * (1 - m)],
    #     #      [0.25 * m ** 2, 0.5 * m * (1 - m), 0.25 * m ** 2]], dtype=torch.float32).cuda()
    #     kerl = kerl.view(1, 1, 3, 3)
    #     x = F.pad(x, (1, 1, 1, 1), 'reflect')
    #     x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
    #     return x_out
    #
    # def H_filter(self, x, m=0.2):
    #     t1 = torch.tensor([[-0.5, 1, -0.5]], dtype=torch.float32).cuda()
    #     t2 = torch.tensor([[0, 1, 0]], dtype=torch.float32).cuda()
    #     t = t1 * m + t2
    #     kerl = torch.matmul(t.T, t)
    #     # kerl = torch.tensor(
    #     #     [[0.25 * m ** 2, -0.5 * m * (1 + m), 0.25 * m ** 2], [-0.5 * m * (1 + m), (1 + m) ** 2, -0.5 * m * (1 + m)],
    #     #      [0.25 * m ** 2, -0.5 * m * (1 + m), 0.25 * m ** 2]], dtype=torch.float32).cuda()
    #     kerl = kerl.view(1, 1, 3, 3)
    #     x = F.pad(x, (1, 1, 1, 1), 'reflect')
    #     x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
    #     return x_out

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

    # def frequency_transform(self, x, n, style='hy', w=None):
    #     if not (w is None) and n == 7:
    #         x1 = self.L_filter(x, w[0])
    #         x2 = self.L_filter(x, w[1])
    #         x3 = self.L_filter(x, w[2])
    #         x4 = self.H_filter(x, w[3])
    #         x5 = self.H_filter(x, w[4])
    #         x6 = self.H_filter(x, w[5])
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #         return x_hy
    #     if not (w is None) and n == 5:
    #         x1 = self.L_filter(x, w[0])
    #         x2 = self.L_filter(x, w[1])
    #         x3 = self.H_filter(x, w[2])
    #         x4 = self.H_filter(x, w[3])
    #         x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
    #         return x_hy
    #     if not (w is None) and n == 3:
    #         x1 = self.L_filter(x, w[0])
    #         x2 = self.H_filter(x, w[1])
    #         x_hy = torch.cat([x, x1, x2], dim=1)
    #         return x_hy
    #
    #     if n == 7 and style == 'hy':
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.L_filter(x, m=0.1)
    #         x3 = self.L_filter(x, m=0.15)
    #         x4 = self.H_filter(x, m=0.05)
    #         x5 = self.H_filter(x, m=0.1)
    #         x6 = self.H_filter(x, m=0.15)
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #     elif n == 7 and style == 'low':
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.L_filter(x, m=0.1)
    #         x3 = self.L_filter(x, m=0.15)
    #         x4 = self.L_filter(x, m=0.2)
    #         x5 = self.L_filter(x, m=0.25)
    #         x6 = self.L_filter(x, m=0.3)
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #     elif n == 7 and style == 'high':
    #         x1 = self.H_filter(x, m=0.05)
    #         x2 = self.H_filter(x, m=0.1)
    #         x3 = self.H_filter(x, m=0.15)
    #         x4 = self.H_filter(x, m=0.2)
    #         x5 = self.H_filter(x, m=0.25)
    #         x6 = self.H_filter(x, m=0.3)
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #     elif n == 3:
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.H_filter(x, m=0.05)
    #         x_hy = torch.cat([x, x1, x2], dim=1)
    #     elif n == 9:
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.L_filter(x, m=0.1)
    #         x3 = self.L_filter(x, m=0.25)
    #         x4 = self.H_filter(x, m=0.05)
    #         x5 = self.H_filter(x, m=0.1)
    #         x6 = self.H_filter(x, m=0.25)
    #         x7 = self.L_filter(x, m=0.3)
    #         x8 = self.H_filter(x, m=0.3)
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8], dim=1)
    #     elif n == 11:
    #         x1 = self.L_filter(x, m=0.025)
    #         x2 = self.L_filter(x, m=0.05)
    #         x3 = self.L_filter(x, m=0.075)
    #         x4 = self.L_filter(x, m=0.1)
    #         x5 = self.H_filter(x, m=0.025)
    #         x6 = self.H_filter(x, m=0.05)
    #         x7 = self.H_filter(x, m=0.075)
    #         x8 = self.H_filter(x, m=0.1)
    #         x9 = self.L_filter(x, m=0.125)
    #         x10 = self.H_filter(x, m=0.125)
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim=1)
    #     elif n == 5 and style == 'low':
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.L_filter(x, m=0.1)
    #         x3 = self.L_filter(x, m=0.15)
    #         x4 = self.L_filter(x, m=0.2)
    #         x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
    #     elif n == 5 and style == 'high':
    #         x1 = self.H_filter(x, m=0.05)
    #         x2 = self.H_filter(x, m=0.1)
    #         x3 = self.H_filter(x, m=0.15)
    #         x4 = self.H_filter(x, m=0.2)
    #         x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
    #     elif n == 5 and style == 'hy':
    #         x1 = self.L_filter(x, m=0.05)
    #         x2 = self.L_filter(x, m=0.1)
    #         x3 = self.H_filter(x, m=0.05)
    #         x4 = self.H_filter(x, m=0.1)
    #         x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
    #     else:
    #         x_hy = x
    #     return x_hy

    def frequency_transform(self, x):
        # x1 = self.L_filter(x, m=0.05)
        # x2 = self.H_filter(x, m=0.05)
        # x3 = self.L_filter(x, m=0.1)
        # x4 = self.H_filter(x, m=0.1)
        # x_hy = torch.cat([x, x1, x2, x3, x4], dim=1)
        x1 = self.L_filter(x, m=0.05)
        x2 = self.H_filter(x, m=0.05)
        x_hy = torch.cat([x, x1, x2], dim=1)
        return x_hy

    # def frequency_transform_Multi(self, x, n1=7, n2=3, w1=None, w2=None):
    #     if n1 == 7:
    #         x1 = self.L_filter(x, m=self.w_last[0])
    #         x1 = self.Multi1(self.frequency_transform(x1, n2, w=self.w_Multi1))
    #         x2 = self.L_filter(x, m=self.w_last[1])
    #         x2 = self.Multi2(self.frequency_transform(x2, n2, w=self.w_Multi2))
    #         x3 = self.L_filter(x, m=self.w_last[2])
    #         x3 = self.Multi3(self.frequency_transform(x3, n2, w=self.w_Multi3))
    #         x4 = self.H_filter(x, m=self.w_last[3])
    #         x4 = self.Multi4(self.frequency_transform(x4, n2, w=self.w_Multi4))
    #         x5 = self.H_filter(x, m=self.w_last[4])
    #         x5 = self.Multi5(self.frequency_transform(x5, n2, w=self.w_Multi5))
    #         x6 = self.H_filter(x, m=self.w_last[5])
    #         x6 = self.Multi6(self.frequency_transform(x6, n2, w=self.w_Multi6))
    #         x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #     if n1 == 3:
    #         x1 = self.frequency_transform(x, n2)
    #         x2 = self.frequency_transform(x1, n2)
    #         x_hy = self.frequency_transform(x2, n2)
    #     if n1 == 5:
    #         w_Multi1 = self.tanh(w2[0]) * 0.2
    #         w_Multi2 = self.tanh(w2[1]) * 0.2
    #         w_Multi3 = self.tanh(w2[2]) * 0.2
    #         w_Multi4 = self.tanh(w2[3]) * 0.2
    #         w_Multi5 = self.tanh(w2[4]) * 0.2
    #         x1 = self.L_filter(x, m=w1[0])
    #         x1 = self.frequency_transform(x1, n2, w=w_Multi1)
    #         x2 = self.L_filter(x, m=w1[1])
    #         x2 = self.frequency_transform(x2, n2, w=w_Multi2)
    #         x3 = self.L_filter(x, m=w1[2])
    #         x3 = self.frequency_transform(x3, n2, w=w_Multi3)
    #         x4 = self.L_filter(x, m=w1[3])
    #         x4 = self.frequency_transform(x4, n2, w=w_Multi4)
    #         x5 = self.frequency_transform(x, n2, w=w_Multi5)
    #         x_hy = torch.cat([x1, x2, x3, x4, x5], dim=1)
    #     # if n1 == 7:
    #     #     x1 = self.L_filter(x, m=0.05)
    #     #     x1 = self.Multi1(self.frequency_transform(x1, n2))
    #     #     x2 = self.L_filter(x, m=0.1)
    #     #     x2 = self.Multi2(self.frequency_transform(x2, n2))
    #     #     x3 = self.L_filter(x, m=0.15)
    #     #     x3 = self.Multi3(self.frequency_transform(x3, n2))
    #     #     x4 = self.H_filter(x, m=0.05)
    #     #     x4 = self.Multi4(self.frequency_transform(x4, n2))
    #     #     x5 = self.H_filter(x, m=0.1)
    #     #     x5 = self.Multi5(self.frequency_transform(x5, n2))
    #     #     x6 = self.H_filter(x, m=0.15)
    #     #     x6 = self.Multi6(self.frequency_transform(x6, n2))
    #     #     x_hy = torch.cat([x, x1, x2, x3, x4, x5, x6], dim=1)
    #     return x_hy

    # def patch_frequency_transform(self, x, style, patch_size=4, n=11):
    #     _, _, h, w = x.shape
    #     row = int(h / patch_size ** (0.5))
    #     col = int(h / patch_size ** (0.5))
    #     patch1 = x[:, :, :row, :col]
    #     patch2 = x[:, :, :row, col:]
    #     patch3 = x[:, :, row:, :col]
    #     patch4 = x[:, :, row:, col:]
    #     if style == 'low':
    #         patch1 = self.patch_convL1(self.frequency_transform(patch1, n))
    #         patch2 = self.patch_convL2(self.frequency_transform(patch2, n))
    #         patch3 = self.patch_convL3(self.frequency_transform(patch3, n))
    #         patch4 = self.patch_convL4(self.frequency_transform(patch4, n))
    #     elif style == 'high':
    #         patch1 = self.patch_convH1(self.frequency_transform(patch1, n))
    #         patch2 = self.patch_convH2(self.frequency_transform(patch2, n))
    #         patch3 = self.patch_convH3(self.frequency_transform(patch3, n))
    #         patch4 = self.patch_convH4(self.frequency_transform(patch4, n))
    #     output = torch.cat((torch.cat((patch1, patch2), dim=3),
    #                         torch.cat((patch3, patch4), dim=3)), dim=2)
    #     return output
    def FAT_low(self, x):
        b, c, h, w = x.shape
        x_sub = x[:, :, ::6, ::6]
        A = self.SA_low(x_sub)
        weight = A.reshape(b, c ** 2, 1, 1)
        x = torch.cat([x] * c, dim=1)
        y = (x * weight).reshape(b, c, c, h, w)
        y = torch.sum(y, dim=2)
        y = torch.sum(y.reshape(b, 1, c, h, w), dim=2) / c
        return y

    def FAT_high(self, x):
        b, c, h, w = x.shape
        x_sub = x[:, :, ::6, ::6]
        A = self.SA_high(x_sub)
        weight = A.reshape(b, c ** 2, 1, 1)
        x = torch.cat([x] * c, dim=1)
        y = (x * weight).reshape(b, c, c, h, w)
        y = torch.sum(y, dim=2)
        y = torch.sum(y.reshape(b, 1, c, h, w), dim=2) / c
        return y

    def FAT(self, x, x_copy):
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

    # def fat_gt(self, x):
    #     b, c, h, w = x.shape
    #     x_sub = x[:, :, ::4, ::4]
    #     A = self.SA_gt(x_sub)[:, :, 0:1, :].reshape(b, c, 1, 1)
    #     y = torch.sum(x * A, dim=1).view(b, 1, h, w)
    #     return y

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # if not (y is None):
        #     low_gt = torch.cat([self.L_filter(y, 0.4), self.L_filter(y, 0.8)], dim=1)
        #     gt_low = self.fat_gt(low_gt)
        #     gt_high = y - gt_low
        # else:
        #     gt_low = None
        #     gt_high = None
        if self.upsampler == 'pixelshuffle':
            # # # stage 1:
            # # x_shallow = self.shallow_feature(x)
            #
            # # stage 2:
            # x = self.conv_first(x)
            # x = self.conv_after_body(self.forward_features(x)) + x
            # x = self.conv_before_upsample(x)
            # x = self.conv_last(self.upsample(x))
            # # x = self.low_high_conv(self.frequency_transform(x))
            # # x = self.conv_last1(x_pr)
            # # x = self.Gauss_filter(x)
            # # x = self.mlp_last(x_inv) + self.Gauss_filter(x_inv)

            # stage 1:
            x_shallow = self.conv_first(x)
            x = self.Shared_layers(x_shallow)

            # stage 2: low
            x_low = self.conv_after_body_low(self.forward_features_low(x, x_shallow)) + x_shallow
            x_low = self.conv_before_upsample_low(x_low)
            x_low = self.conv_last_low(self.upsample_low(x_low))
            # x_low = self.frequency_transform(self.frequency_transform(x_low))
            # x_low = self.frequency_transform(x_low)
            # x_low = self.FAT_low(x_low)

            # x_low = self.frequency_transform_Multi(x_low, n1=5, n2=3, w1=self.tanh(self.w_low)*0.2,
            #                                        w2=self.w_list_low)
            # x_low1 = self.frequency_transform(x_low[:, 0:1, :, :], n=5, w=self.tanh(self.w_low1)*0.2)
            # x_low2 = self.frequency_transform(x_low[:, 1:2, :, :], n=5, w=self.tanh(self.w_low2)*0.2)
            # x_low3 = self.frequency_transform(x_low[:, 2:3, :, :], n=5, w=self.tanh(self.w_low3)*0.2)
            # x_low = self.low_conv(self.FAT_low(self.frequency_transform(x_low)))
            # x_low = self.frequency_transform_Multi(x_low, n1=3, n2=3)
            # x_low = self.FAT_low(x_low)
            # x_low = self.low_Conv()

            # x_lowhy = self.low_high_conv(self.frequency_transform(x_low, n=7, w=self.tanh(self.w_low)*0.2))
            # x_lowhy = self.patch_frequency_transform(x_low, 'low')

            # x_lowhy = self.low_high_conv_2(self.frequency_transform(x_lowhy, n=7))
            # x_low = self.fft_low(x_low)

            # stage 2: high1 or low_scale
            x_high = self.conv_after_body_high(self.forward_features_high(x, x_shallow)) + x_shallow
            x_high = self.conv_before_upsample_high(x_high)
            x_high = self.conv_last_high(self.upsample_high(x_high))
            # x_high = self.frequency_transform(self.frequency_transform(x_high))
            # x_high = self.frequency_transform(x_high)
            # x_high = self.FAT_high(x_high)
            # x_high = self.frequency_transform_Multi(x_high, n1=5, n2=3, w1=self.tanh(self.w_high) * 0.2,
            #                                        w2=self.w_list_high)
            # x_high = self.high_conv(self.FAT_high(self.frequency_transform(x_high)))
            # x_high = self.frequency_transform_Multi(x_high, n1=3, n2=3)
            # x_high = self.FAT_high(x_high)
            # x_high = self.high_Conv()
            # x_highhy = self.low_high_conv1(self.frequency_transform(x_high, n=7, w=self.tanh(self.w_high)*0.2))
            # x_highhy = self.patch_frequency_transform(x_high, 'high')

            # x_highhy = self.low_high_conv1_2(self.frequency_transform(x_highhy, n=7))
            # x_high = self.fft_high(x_high)

            # # stage 3: high2
            # x_high1 = self.conv_after_body_high1(self.forward_features_high1(x, x_shallow)) + x_shallow
            # x_high1 = self.conv_before_upsample_high1(x_high1)
            # x_high1 = self.conv_last_high1(self.upsample_high1(x_high1))
            # # x_high = self.fft_high(x_high)
            #
            # # stage 2: high
            # x_high = self.features_get(x_low)
            # b, c = x_high.shape[0], x_high.shape[1]
            # x_high = x_high.permute(0, 2, 3, 1).contiguous().view(b, 64*64, c)
            # x_high = self.mlp_high(x_high)
            # x_high = x_high.view(b, 64, 64, c).permute(0, 3, 1, 2)
            # x_high = self.conv_high(x_high)

            # stage 3: low + high
            # x_high_new = self.Gauss_inv(x_high)
            x = x_low + x_high
            x_copy = x
            # for i in range(3):
            #     x1 = self.L_filter(x, self.weight[i*3])
            #     x2 = self.L_filter(x, self.weight[i*3+1])
            #     x3 = self.L_filter(x, self.weight[i*3+2])
            #     x = torch.mean(torch.cat((x1, x2, x3), dim=1), dim=1).view(x.shape[0], 1, 64, 64)
            x = self.frequency_transform(self.frequency_transform(x))
            x = self.FAT(x, x_copy)

            # x = self.low_high_conv(self.frequency_transform_Multi(x))
            # x_hy = torch.cat([x_lowhy, x_highhy], dim=1)
            # x = self.low_high_conv2(x_hy)
            # x = self.patch_frequency_transform(x)
            # x = self.low_high_conv2_2(self.frequency_transform(x, n=7))
            #
            # # stage 3: low + high
            # x = self.fft_last(x_low, x_high)
            #
            # # stage 3: low + high
            # x = x_low + x_high
            # 笑死

        x = x / self.img_range + self.mean

        # return (x, x), x
        return (x_low, x_high), x