import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.fft as fft
import numpy as np
from collections import OrderedDict
import tifffile
import os
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img, get_root_logger
from models.DEMfeature_model import DEMFeature

import math
from tqdm import tqdm
from os import path as osp

from RiverModel.unet.unet_model import UNet as RiverNet  # unet

class RiverLoss(nn.Module):
    def __init__(self, extract_river_Weights):
        super(RiverLoss, self).__init__()
        self.unet = RiverNet(n_channels=1, n_classes=1)  # unet
        self.unet.load_state_dict(torch.load(extract_river_Weights))
    def forward(self, input, targets):

        fake_river_heatmap = self.unet(input)
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(fake_river_heatmap, targets)

        return loss

@MODEL_REGISTRY.register()
class HATModel(SRModel):

    def get_unet(self):
        unet = RiverNet(n_channels=1, n_classes=1)  # unet
        unet.load_state_dict(torch.load('/home/ch/HAT/unet.pth'))
        unet.to(self.device)
        river_conterion = RiverLoss('/home/ch/HAT/unet.pth')
        river_conterion.to(self.device)
        return unet, river_conterion
    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output = self.net_g_ema(self.img)
                self.output1, self.output = self.net_g_ema(self.img)
                # self.output = self.output[1]
        else:
            self.net_g.eval()
            with torch.no_grad():
                # self.output = self.net_g(self.img)
                self.output1, self.output = self.net_g(self.img)
                # self.output = self.output[1]
            # self.net_g.train()
    # ---------------------------此处开始修改1--------------------------------------------

    def high_pass_filter_tensor(self, input_tensor, dim):
        """
        对指定维度进行高通滤波的函数，使用傅里叶变换

        Parameters:
        input_tensor (torch.Tensor): 输入张量
        dim (int): 指定的维度
        cutoff_frequency (float): 截止频率，高频信息保留的阈值

        Returns:
        torch.Tensor: 高通滤波后的张量
        """
        # 进行傅里叶变换
        f_transform = fft.fftn(input_tensor, dim=dim)
        f_transform_shifted = fft.fftshift(f_transform, dim=dim)

        # 获取张量尺寸
        dims = input_tensor.shape
        dims_list = list(dims)

        # 构造高通滤波器
        s = 4
        rows, cols = dims_list[2], dims_list[3]
        crow, ccol = int(rows / 2), int(cols / 2)
        f_transform_shifted[:, :, crow - s:crow + s, ccol - s:ccol + s] = 0

        # 进行逆傅里叶变换
        f_transform = fft.fftshift(f_transform_shifted, dim=dim)
        output_tensor = fft.ifftn(f_transform, dim=dim)
        output_tensor = torch.abs(output_tensor)

        return output_tensor

    def get_fft(self, x):
        x_fft = torch.fft.rfft2(x, norm='backward')
        x_amp = torch.abs(x_fft).cuda()
        x_pha = torch.angle(x_fft).cuda()
        return x_amp, x_pha

    # def cal_cosloss(self, x, y, scale=4):
    #     b, c, h, w = x.shape
    #     # cos_matrix = torch.zeros(b, c, int(h / scale), int(w / scale)).cuda()
    #     an = torch.zeros(1,1).cuda()
    #     for k in range(b):
    #         for i in range(int(h / scale)):
    #             for j in range(int(w / scale)):
    #                 s1 = x[k, :, scale * i:scale * i + scale, scale * j:scale * j + scale].reshape(1, scale ** 2)
    #                 s2 = y[k, :, scale * i:scale * i + scale, scale * j:scale * j + scale].reshape(1, scale ** 2)
    #                 s1_s2 = torch.matmul(s1, s2.T)
    #                 s1_abs = torch.sqrt(torch.matmul(s1, s1.T)) + 1e-8
    #                 s2_abs = torch.sqrt(torch.matmul(s2, s2.T)) + 1e-8
    #                 # cos_matrix[k, :, i, j] = s1_s2 / (s1_abs * s2_abs)
    #                 an = an - (s1_s2 / (s1_abs * s2_abs) + 1)
    #     an = torch.mean(an)
    #     out = scale**2*an/(b*h*w)
    #     # t = -(cos_matrix + 1)
    #     # out = torch.mean(t)
    #     return out
    def cal_thetaloss(self, x, y, scale=4):
        b, c, h, w = x.shape
        cos_matrix = torch.zeros(b, c, int(h / scale), int(w / scale)).cuda()
        for i in range(int(h / scale)):
            for j in range(int(w / scale)):
                s1 = x[0:b, 0:c, scale * i:scale * i + scale, scale * j:scale * j + scale].reshape(b, c, scale ** 2)
                s2 = y[0:b, 0:c, scale * i:scale * i + scale, scale * j:scale * j + scale].reshape(b, c, scale ** 2)
                cos_matrix[0:b, 0:c, i, j] = F.cosine_similarity(s1, s2, dim=2)
        t = -(cos_matrix + 1)
        out = torch.mean(t)
        return out
    # def get_inv_matrix(self, n=64):
    #     x = torch.zeros(n ** 2, (n + 2) ** 2)
    #     y = torch.zeros((n + 2) ** 2, n ** 2)
    #     for i in range(n):
    #         for j in range(n):
    #             x[i * n + j, i * (n + 2) + j] = 1 / 20
    #             x[i * n + j, i * (n + 2) + j + 1] = 2 / 20
    #             x[i * n + j, i * (n + 2) + j + 2] = 1 / 20
    #
    #             x[i * n + j, i * (n + 2) + j + n + 2] = 2 / 20
    #             x[i * n + j, i * (n + 2) + j + n + 3] = 8 / 20
    #             x[i * n + j, i * (n + 2) + j + n + 4] = 2 / 20
    #
    #             x[i * n + j, i * (n + 2) + j + 2 * n + 4] = 1 / 20
    #             x[i * n + j, i * (n + 2) + j + 2 * n + 5] = 2 / 20
    #             x[i * n + j, i * (n + 2) + j + 2 * n + 6] = 1 / 20
    #     for i in range(n + 2):
    #         for j in range(n + 2):
    #             if i == 0:
    #                 if j == 0:
    #                     y[i * (n + 2) + j, (i + 1) * n + 1] = 1
    #                 elif j == n + 1:
    #                     y[i * (n + 2) + j, (i + 1) * n + n - 2] = 1
    #                 else:
    #                     y[i * (n + 2) + j, (i + 1) * n + j - 1] = 1
    #             elif i == n + 1:
    #                 if j == 0:
    #                     y[i * (n + 2) + j, (i - 3) * n + 1] = 1
    #                 elif j == n + 1:
    #                     y[i * (n + 2) + j, (i - 3) * n + n - 2] = 1
    #                 else:
    #                     y[i * (n + 2) + j, (i - 3) * n + j - 1] = 1
    #             else:
    #                 if j == 0:
    #                     y[i * (n + 2) + j, (i - 1) * n + 1] = 1
    #                 elif j == n + 1:
    #                     y[i * (n + 2) + j, (i - 1) * n + n - 2] = 1
    #                 else:
    #                     y[i * (n + 2) + j, (i - 1) * n + j - 1] = 1
    #     inv = torch.inverse(torch.matmul(x, y))
    #     return inv
    #
    # def Gauss_inv(self, x):
    #     b, c, h, w = x.shape
    #     inv = self.get_inv_matrix().to(self.device)
    #     output = torch.zeros(b, c, h, w).to(self.device)
    #     for i in range(b):
    #         y = x[i, 0:c, :, :]
    #         output[i, 0:c, :, :] = torch.matmul(inv, y.view(64 * 64, 1)).view(64, 64).unsqueeze(0).unsqueeze(0)
    #     return output
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt_path = data['gt_path']
        if 'weight' in data:
            self.weight = data['weight'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'img_pr' in data:
            self.img_pr = data['img_pr'].to(self.device)
        if 'river_weight' in data:
            self.river_weight = data['river_weight'].to(self.device)
        if 'img_gt_low' in data:
            self.img_gt_low = data['img_gt_low'].to(self.device)
        if 'img_gt_low_new' in data:
            self.img_gt_low_new = data['img_gt_low_new'].to(self.device)
        if 'img_gt_low_new1' in data:
            self.img_gt_low_new1 = data['img_gt_low_new1'].to(self.device)
        if 'img_gt_high_new' in data:
            self.img_gt_high_new = data['img_gt_high_new'].to(self.device)
        if 'img_gt_high' in data:
            self.img_gt_high = data['img_gt_high'].to(self.device)
        if 'img_gt_low1' in data:
            self.img_gt_low1 = data['img_gt_low1'].to(self.device)
        if 'img_gt_high1' in data:
            self.img_gt_high1 = data['img_gt_high1'].to(self.device)
        if 'min_scale' in data:
            self.min_scale = data['min_scale'].to(self.device)
        if 'max_scale' in data:
            self.max_scale = data['max_scale'].to(self.device)
        if 'low_scale' in data:
            self.low_scale = data['low_scale'].to(self.device)

    def huber_loss(self, y_true, y_pred, delta=1.5e-2):
        error = torch.abs(y_true - y_pred)
        quadratic_part = torch.clamp(error, 0.0, delta)
        linear_part = error - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + delta * linear_part
        return loss.mean()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output1, self.output2= self.net_g(self.lq)
        # if current_iter % 6000 == 0:
        #     print(self.net_g.module.w_low)
        # criterion = torch.nn.SmoothL1Loss()
        # self.output2 = self.net_g(self.lq)
        # if torch.isnan(self.output2).any():
        #     print(current_iter)
        #     print(self.output2)
        # assert not torch.isnan(self.output2).any(), "NaN detected!"
        # if current_iter % 4000 == 0:
        #     for name, param in self.net_g.named_parameters():
        #         # if current_iter % 50 == 0 and name == 'module.high_frequency_Grad_Intermingling.pre4.weight':
        #         #     print(param)
        #         assert not torch.isnan(param).any(), "NaN detected!"


        l_total = 0
        loss_dict = OrderedDict()
        # unet, river_conterion = self.get_unet()
        # high_river = 1.0 * (F.sigmoid(unet(self.gt)) > 0.5).detach().cpu().numpy().astype(np.float32)
        # high_river_heatmap = torch.tensor(high_river).cuda()
        # generator_river_loss = river_conterion(self.output2, high_river_heatmap)
        # l_total += generator_river_loss * 0.001
        # loss_dict['generator_river_loss'] = generator_river_loss

        # pixel loss 重建损失
        # output2的重建损失
        if self.cri_pix:
            # l_pix2 = self.huber_loss(self.output2, self.gt)
            # l_pix2 = torch.nn.functional.l1_loss(self.output2*self.river_weight, self.gt*self.river_weight)
            l_pix2 = torch.nn.functional.l1_loss(self.output2, self.gt)
            # l_pix2 = (0.5*torch.exp(torch.abs(self.output2 - self.gt)*0.45) - 0.5*torch.exp(-torch.abs(self.output2 - self.gt)*0.45)).mean()
            # l_pix2 = torch.log(torch.abs(self.output2 - self.gt) + 1).mean()
            # if current_iter < 40000:
            #     l_pix2 = torch.nn.functional.l1_loss(self.output2, self.gt)
            # else:
            #     l_pix2 = self.cri_pix(self.output2, self.gt)
            l_total += l_pix2
            loss_dict['l_pix2'] = l_pix2
            # l_cos = self.cal_thetaloss(self.output2, self.gt)
            # l_total += l_cos
            # loss_dict['l_cos'] = l_cos
            # l_pix2 = torch.mean(torch.sqrt(torch.abs(self.output2 - self.gt)))
            # l_pix2 = criterion(self.output2, self.gt)
            # l_total += l_pix2
            # loss_dict['l_pix2'] = l_pix2

            # if torch.isnan(l_total).any():
            #     print(current_iter)
            #     print('output:', self.output2)
            #     print('---'*10)
            #     print('gt:', self.gt)
            # # print(current_iter, l_total)
            # assert not torch.isnan(l_total).any(), "NaN detected!"
        #     l_pix2_1 = torch.nn.functional.l1_loss(self.output2 * self.weight, self.gt * self.weight)
        #     l_total += l_pix2_1 * 0.5
        #     loss_dict['l_pix2_1'] = l_pix2_1
        #     # l_pix1 = self.cri_pix(self.output1 * self.river_weight, self.gt * self.river_weight)
        #     # l_total += l_pix1 * 0.5 * 0.5
        #     # loss_dict['l_pix1'] = l_pix1
        #     # print(l_total)
        #     # print(self.output2)
        #     # print(self.gt)
        #     if torch.isnan(l_total).any():
        #         print(current_iter)
        #     assert not torch.isnan(l_total).any(), "NaN detected!"

        # output1的重建损失
        if self.cri_pix:
            # l_pix1_0 = self.huber_loss(self.output1[0], self.img_gt_low, 1e-2)
            # l_pix1_1 = self.huber_loss(self.output1[1], self.img_gt_high, 5e-3)
            # l_pix1_0 = torch.nn.functional.l1_loss(self.output1[0]*self.river_weight, self.img_gt_low*self.river_weight)
            # l_pix1_1 = torch.nn.functional.l1_loss(self.output1[1]*self.river_weight, self.img_gt_high*self.river_weight)
            l_pix1_0 = torch.nn.functional.l1_loss(self.output1[0], self.img_gt_low)
            l_pix1_1 = torch.nn.functional.l1_loss(self.output1[1], self.img_gt_high)
            # l_pix1_0 = (0.5*torch.exp(torch.abs(self.output1[0] - self.img_gt_low)*0.45) - 0.5*torch.exp(-torch.abs(self.output1[0] - self.img_gt_low)*0.45)).mean()
            # l_pix1_1 = (0.5*torch.exp(torch.abs(self.output1[1] - self.img_gt_high)*0.45) - 0.5*torch.exp(-torch.abs(self.output1[1] - self.img_gt_high)*0.45)).mean()
            # l_pix1_0 = torch.log(torch.abs(self.output1[0] - self.img_gt_low) + 1).mean()
            # l_pix1_1 = torch.log(torch.abs(self.output1[1] - self.img_gt_high) + 1).mean()
            # if current_iter < 40000:
            #     l_pix1_0 = torch.nn.functional.l1_loss(self.output1[0], self.img_gt_low)
            #     l_pix1_1 = torch.nn.functional.l1_loss(self.output1[1], self.img_gt_high)
            # else:
            #     l_pix1_0 = self.cri_pix(self.output1[0], self.img_gt_low)
            #     l_pix1_1 = self.cri_pix(self.output1[1], self.img_gt_high)
            l_total += l_pix1_0
            loss_dict['l_pix1_0'] = l_pix1_0
            # 笑死
            # l_pix1_1 = torch.nn.functional.l1_loss(self.output1[1]*self.river_weight, self.img_gt_high*self.river_weight)
            l_total += l_pix1_1
            loss_dict['l_pix1_1'] = l_pix1_1
            # # l_pix1_1 = self.cri_pix(self.output1[1], self.img_gt_low1)
            # l_pix1_2 = torch.nn.functional.l1_loss(self.output1[2], self.img_gt_high_new)
            # l_total += l_pix1_2
            # loss_dict['l_pix1_2'] = l_pix1_2

        #     l_pix1_1_r = torch.nn.functional.l1_loss(self.output1[1] * self.river_weight, self.img_gt_high * self.river_weight)
        #     l_total += l_pix1_1_r * 0.005
        #     loss_dict['l_pix1_1_r'] = l_pix1_1_r
        #     if torch.isnan(l_total).any():
        #         if torch.isnan(self.low_scale).any():
        #             print(self.gt_path)
        #     assert not torch.isnan(l_total).any(), "NaN detected!"
        #     # l_pix1_2 = self.cri_pix(self.output1[2], self.img_gt_high1)
        #     # l_total += l_pix1_2
        #     # loss_dict['l_pix1_2'] = l_pix1_2

        # output2的梯度损失
        if current_iter > 8000 and False:
            kerlx = torch.tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
            kerly = torch.tensor([[0, 1, 0], [0, 0, 0], [0, -1, 0]], dtype=torch.float32).view(1, 1, 3, 3).cuda()
            gt = F.pad(self.gt, (1, 1, 1, 1), 'reflect')
            gt_x = F.conv2d(input=gt, weight=kerlx, bias=None, stride=1, padding=0, dilation=1, groups=1)
            gt_y = F.conv2d(input=gt, weight=kerly, bias=None, stride=1, padding=0, dilation=1, groups=1)
            output2 = F.pad(self.output2, (1, 1, 1, 1), 'reflect')
            outx = F.conv2d(input=output2, weight=kerlx, bias=None, stride=1, padding=0, dilation=1, groups=1)
            outy = F.conv2d(input=output2, weight=kerly, bias=None, stride=1, padding=0, dilation=1, groups=1)
            # #Scharr 算子
            # scharr_x = torch.tensor([[3/16, 0, -3/16], [10/16, 0, -10/16], [3/16, 0, -3/16]], dtype=torch.float32).cuda()
            # scharr_y = torch.tensor([[3/16, 10/16, 3/16], [0, 0, 0], [-3/16, -10/16, -3/16]], dtype=torch.float32).cuda()
            # scharr_x = scharr_x.view(1, 1, 3, 3)  # 适应卷积操作的形状
            # scharr_y = scharr_y.view(1, 1, 3, 3)  # 适应卷积操作的形状

            # # 应用 Scharr 算子
            # gt_grad_x = F.conv2d(self.gt, scharr_x, padding=1)
            # gt_grad_y = F.conv2d(self.gt, scharr_y, padding=1)
            # output1_grad_x = F.conv2d(self.output1[0], scharr_x, padding=1)
            # output1_grad_y = F.conv2d(self.output1[0], scharr_y, padding=1)

            l_grad2 = F.mse_loss(gt_x, outx) + F.mse_loss(gt_y, outy)
            l_total += l_grad2 * 0.005
            # assert not torch.isnan(l_total).any(), "NaN detected!"
            loss_dict['l_grad2'] = l_grad2

        # # output1的高频损失
        # if 2:
        #     gt_fr = self.high_pass_filter_tensor(input_tensor=self.gt, dim=(2, 3))
        #     output1_fr = self.high_pass_filter_tensor(input_tensor=self.output1, dim=(2, 3))
        #     l_fre1 = F.mse_loss(gt_fr, output1_fr)
        #     l_total += l_fre1 * 0.1
        #     loss_dict['l_fre1'] = l_fre1

        # output1的fft损失
        if current_iter > 10000 and False:
            gt_fft = self.get_fft(self.gt)
            gt_amp = gt_fft[0]
            gt_pha = gt_fft[1]
            output1_fft = self.get_fft(self.output1)
            output1_amp = output1_fft[0]
            output1_pha = output1_fft[1]
            l_fft1 = F.mse_loss(gt_amp, output1_amp) + 0.1 * F.mse_loss(gt_pha, output1_pha)
            l_total += l_fft1 * 0.2
            loss_dict['l_fft1'] = l_fft1

        # output2的梯度损失
        if current_iter > 8000 and False:
            # Scharr 算子
            scharr_x = torch.tensor([[3/16, 0, -3/16], [10/16, 0, -10/16], [3/16, 0, -3/16]], dtype=torch.float32).cuda()
            scharr_y = torch.tensor([[3/16, 10/16, 3/16], [0, 0, 0], [-3/16, -10/16, -3/16]], dtype=torch.float32).cuda()
            scharr_x = scharr_x.view(1, 1, 3, 3)  # 适应卷积操作的形状
            scharr_y = scharr_y.view(1, 1, 3, 3)  # 适应卷积操作的形状

            # 应用 Scharr 算子
            gt_grad_x = F.conv2d(self.gt, scharr_x, padding=1)
            gt_grad_y = F.conv2d(self.gt, scharr_y, padding=1)
            output2_grad_x = F.conv2d(self.output2, scharr_x, padding=1)
            output2_grad_y = F.conv2d(self.output2, scharr_y, padding=1)

            l_grad2 = F.mse_loss(gt_grad_x, output2_grad_x) + F.mse_loss(gt_grad_y, output2_grad_y)
            l_total += l_grad2* 0.05
            loss_dict['l_grad2'] = l_grad2

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        assert not torch.isnan(l_total).any(), "NaN detected!"
        l_total.backward()
        self.optimizer_g.step()
        # for name, param in self.net_g.named_parameters():
        #     if current_iter % 100 == 0 and name == 'w_low':
        #         print(param)
        #     # if torch.isnan(param).any():
        #     #     print(current_iter, name)
        #     assert not torch.isnan(param).any(), "NaN detected!"
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # ---------------------------此处结束修改1--------------------------------------------
    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        # out_dict['result_before'] = self.copy.detach().cpu()
        out_dict['low_gt'] = self.img_gt_low.detach().cpu()
        out_dict['high_gt'] = self.img_gt_high.detach().cpu()
        out_dict['low'] = self.output1[0].detach().cpu()
        out_dict['high'] = self.output1[1].detach().cpu()
        # out_dict['min_low'] = self.min_low.detach().cpu()
        # out_dict['max_low'] = self.max_low.detach().cpu()

        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def get_inv_matrix(self, n=64, m=0.27406862):
        m1 = 1 - 2 * m
        a, b, c = m1 ** 2, m * m1, m ** 2
        x = torch.zeros(n ** 2, (n + 2) ** 2)
        y = torch.zeros((n + 2) ** 2, n ** 2)
        z = torch.eye(n**2)
        for i in range(n):
            for j in range(n):
                x[i * n + j, i * (n + 2) + j] = c
                x[i * n + j, i * (n + 2) + j + 1] = b
                x[i * n + j, i * (n + 2) + j + 2] = c

                x[i * n + j, i * (n + 2) + j + n + 2] = b
                x[i * n + j, i * (n + 2) + j + n + 3] = a
                x[i * n + j, i * (n + 2) + j + n + 4] = b

                x[i * n + j, i * (n + 2) + j + 2 * n + 4] = c
                x[i * n + j, i * (n + 2) + j + 2 * n + 5] = b
                x[i * n + j, i * (n + 2) + j + 2 * n + 6] = c
        for i in range(n + 2):
            for j in range(n + 2):
                if i == 0:
                    if j == 0:
                        y[i * (n + 2) + j, (i + 1) * n + 1] = 1
                    elif j == n + 1:
                        y[i * (n + 2) + j, (i + 1) * n + n - 2] = 1
                    else:
                        y[i * (n + 2) + j, (i + 1) * n + j - 1] = 1
                elif i == n + 1:
                    if j == 0:
                        y[i * (n + 2) + j, (i - 3) * n + 1] = 1
                    elif j == n + 1:
                        y[i * (n + 2) + j, (i - 3) * n + n - 2] = 1
                    else:
                        y[i * (n + 2) + j, (i - 3) * n + j - 1] = 1
                else:
                    if j == 0:
                        y[i * (n + 2) + j, (i - 1) * n + 1] = 1
                    elif j == n + 1:
                        y[i * (n + 2) + j, (i - 1) * n + n - 2] = 1
                    else:
                        y[i * (n + 2) + j, (i - 1) * n + j - 1] = 1
        inv = torch.inverse(z + torch.matmul(x, y))
        return inv

    def gauss_inv(self, x, inv):
        x = torch.matmul(inv, x.view(64 * 64, 1)).view(64, 64).numpy()
        return 2 * x
    def Gauss_filter(self, x):
        kerl = torch.tensor([[1 / 20, 2 / 20, 1 / 20], [2 / 20, 8 / 20, 2 / 20], [1 / 20, 2 / 20, 1 / 20]],
                            dtype=torch.float32)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return x_out.view(64, 64).numpy()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if current_iter>74000 and True:
           # pass
        # if False:
        #     pass
        # if (current_iter == 2000 or current_iter == 78000 or current_iter == 80000 or current_iter == 118000 or current_iter == 120000
        #         or current_iter == 156000 or current_iter == 158000 or current_iter == 196000
        #         or current_iter == 198000 or current_iter == 234000 or current_iter == 236000):

            dataset_name = dataloader.dataset.opt['name']
            with_metrics = self.opt['val'].get('metrics') is not None
            use_pbar = self.opt['val'].get('pbar', False)

            if with_metrics:
                if not hasattr(self, 'metric_results'):  # only execute in the first run
                    self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
                # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
                self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            if with_metrics:
                self.metric_results = {metric: 0 for metric in self.metric_results}

            metric_data = dict()
            if use_pbar:
                pbar = tqdm(total=len(dataloader), unit='image')

            rmse_dem_sum = 0
            # rmse_dem_sum1 = 0
            rmse_slope_sum = 0
            rmse_aspect_sum = 0
            # rmse_dem_cell_sum = 0
            j = 0
            for idx, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
                self.feed_data(val_data)
                # self.river_weight = val_data['river_weight'].to(self.device)

                self.pre_process()
                if 'tile' in self.opt:
                    self.tile_process()
                else:
                    self.process()
                self.post_process()

                visuals = self.get_current_visuals()
                if self.opt['dem_tif']:
                    # 反归一化：
                    gt_tif = visuals['gt'].squeeze().numpy()  # (64,64)
                    sr_tif = visuals['result'].squeeze().numpy()  # (64,64)
                    # sr_before = visuals['result_before'].squeeze().numpy()  # (64,64)
                    # low_tif = visuals['low'].squeeze().numpy()
                    h = sr_tif.shape[-1]
                    # sr_tif1 = visuals['result']
                    # sr_tif = self.gauss_inv(sr_tif1, inv)
                    # sr_tif = self.Gauss_filter(sr_tif)
                    # low = visuals['low'].squeeze().numpy()  # (64,64)
                    # # sr_tif = low
                    # high = visuals['high'].squeeze().numpy()  # (64,64)
                    # low_gt = visuals['low_gt'].squeeze().numpy()  # (64,64)
                    # high_gt = visuals['high_gt'].squeeze().numpy()  # (64,64)
                    max_tif = val_data['max'].numpy()
                    min_tif = val_data['min'].numpy()

                    sr_tif = ((sr_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    # sr_before = ((sr_before + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    # low_tif = ((low_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif

                    gt_tif = ((gt_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    dem_feature = DEMFeature(torch.tensor(sr_tif).reshape(1,1,h,h),torch.tensor(gt_tif).reshape(1,1,h,h))
                    slope = dem_feature['slope_1_rmse']
                    aspect = dem_feature['aspect_1_rmse']
                    # 计算平方误差
                    squared_error = (sr_tif - gt_tif) ** 2
                    # squared_error1 = (sr_before - gt_tif) ** 2
                    # squared_error2 = (low_tif - gt_tif) ** 2
                    # squared_error_low = (low - low_gt) ** 2
                    # squared_error_high = (high - high_gt) ** 2

                    # 计算均方根误差
                    mean_squared_error = np.mean(squared_error)
                    rmse_dem = np.sqrt(mean_squared_error)
                    # rmse_dem_cell = np.sqrt(np.mean((sr_tif - gt_tif)[4:-4, 4:-4] ** 2))
                    # mean_squared_error1 = np.mean(squared_error1)
                    # rmse_dem1 = np.sqrt(mean_squared_error1)

                    # mean_squared_error2 = np.mean(squared_error2)
                    # rmse_dem2 = np.sqrt(mean_squared_error2)
                    # mean_squared_error_low = np.mean(squared_error_low)
                    # rmse_dem_low = np.sqrt(mean_squared_error_low)
                    # mean_squared_error_high = np.mean(squared_error_high)
                    # rmse_dem_high = np.sqrt(mean_squared_error_high)
                    if False:
                        errlog = open('./EachDem-rmse.txt', 'a')
                        errlog.write(img_name + ','  + str(rmse_dem1) + ',' +str(rmse_dem))
                        errlog.write('\n')
                        errlog.close()
                        # save_tif_path = osp.join(self.opt['path']['visualization'], dataset_name,
                        #                          f'{img_name}.TIF')
                        # output_folder = osp.join(self.opt['path']['visualization'], dataset_name)
                        # os.makedirs(output_folder, exist_ok=True)
                        # tifffile.imwrite(save_tif_path, sr_tif)

                    # print(f'{img_name}_rmse_dem:', rmse_dem)
                    # print(f'{img_name}_rmse_dem_low:', rmse_dem_low)
                    # print(f'{img_name}_rmse_dem_high:', rmse_dem_high)

                    # save_tif_path1 = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                          f'{img_name}low.TIF')
                    # save_tif_path2 = osp.join(self.opt['path']['visualization'], dataset_name,
                    #                          f'{img_name}high.TIF')

                    # tifffile.imwrite(save_tif_path1, low)
                    # tifffile.imwrite(save_tif_path2, high)

                    rmse_dem_sum = rmse_dem + rmse_dem_sum
                    # rmse_dem_sum1 = rmse_dem1 + rmse_dem_sum1

                    rmse_slope_sum = slope + rmse_slope_sum
                    rmse_aspect_sum = aspect + rmse_aspect_sum
                    # rmse_dem_cell_sum = rmse_dem_cell + rmse_dem_cell_sum
                    j = j + 1
                    # print(f'{j}完成！')
                    # print('rmse_dem_sum:',rmse_dem_sum)
                    # print('-----------------------------------------')

                    # min = np.min(img_gt)
                    # max = np.max(img_gt)
                    # img_gt = 2 * (img_gt - min) / (max - min + 10) - 1
                    # img_lq = 2 * (img_lq - min) / (max - min + 10) - 1
                else:
                    sr_img = tensor2img([visuals['result']])
                    metric_data['img'] = sr_img
                    if 'gt' in visuals:
                        gt_img = tensor2img([visuals['gt']])
                        metric_data['img2'] = gt_img
                        del self.gt

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if save_img and not self.opt['dem_tif']:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_img_path)

                # sr_img = tensor2img([visuals['result']])
                # metric_data['img'] = sr_img
                # if 'gt' in visuals:
                #     gt_img = tensor2img([visuals['gt']])
                #     metric_data['img2'] = gt_img
                #     del self.gt
                #
                # # tentative for out of GPU memory
                # del self.lq
                # del self.output
                # torch.cuda.empty_cache()
                #
                # if save_img:
                #     if self.opt['is_train']:
                #         save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                #                                  f'{img_name}_{current_iter}.png')
                #     else:
                #         if self.opt['val']['suffix']:
                #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                #                                      f'{img_name}_{self.opt["val"]["suffix"]}.png')
                #         else:
                #             save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                #                                      f'{img_name}_{self.opt["name"]}.png')
                #     imwrite(sr_img, save_img_path)

                if with_metrics and not self.opt['dem_tif']:
                    # calculate metrics
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
                elif with_metrics and self.opt['dem_tif']:
                    # calculate metrics of dem
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

                if use_pbar:
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')

            # print('the m-rmse of dem is :', rmse_dem_sum / j)
            # # print('the before_m-rmse of dem is :', rmse_dem_sum1 / j)
            # print('the slope-rmse of dem is :', rmse_slope_sum / j)
            # print('the aspect-rmse of dem is :', rmse_aspect_sum / j)

            # print('the m-rmse_cell of dem is :', rmse_dem_cell_sum / j)
            rmse_dem_mean = rmse_dem_sum / j
            rmse_slope_mean = rmse_slope_sum / j
            rmse_aspect_mean = rmse_aspect_sum / j
            logger = get_root_logger()
            logger.info(f'the m-rmse of dem is :{rmse_dem_mean}')
            logger.info(f'the slope-rmse of dem is :{rmse_slope_mean}')
            logger.info(f'the aspect-rmse of dem is :{rmse_aspect_mean}')

            # errlog = open('./Dem_test.txt', 'a')
            # errlog.write(str(current_iter) + ',' + str(rmse_dem_mean))
            # errlog.write('\n')
            # errlog.write(str(current_iter) + ',' + str(rmse_slope_mean))
            # errlog.write('\n')
            # errlog.write(str(current_iter) + ',' + str(rmse_aspect_mean))
            # errlog.write('\n')
            # errlog.close()
            if use_pbar:
                pbar.close()

            if with_metrics:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= (idx + 1)
                    # update the best metric result
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def save(self, epoch, current_iter):
        # if (current_iter==40000 or current_iter==42000 or current_iter==44000 or current_iter==62000
        #         or current_iter==64000 or current_iter==66000 or current_iter==82000
        #         or current_iter==84000 or current_iter==86000 or current_iter==104000
        #         or current_iter==106000 or current_iter==108000 or current_iter==126000
        #         or current_iter==128000):
        #     if hasattr(self, 'net_g_ema'):
        #         self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,
        #                           param_key=['params', 'params_ema'])
        #     else:
        #         self.save_network(self.net_g, 'net_g', current_iter)
        #     self.save_training_state(epoch, current_iter)
        # else:
        #     pass
        # if (current_iter==10000 or current_iter==14000 or current_iter==16000 or current_iter==18000
        #         or current_iter==20000 or current_iter==8000 or current_iter==12000
        #         or current_iter==26000 or current_iter==24000 or current_iter==28000):
        #     if hasattr(self, 'net_g_ema'):
        #         self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,
        #                           param_key=['params', 'params_ema'])
        #     else:
        #         self.save_network(self.net_g, 'net_g', current_iter)
        #     self.save_training_state(epoch, current_iter)
        # else:
        #     pass
        if (current_iter == 78000 or current_iter == 80000 or current_iter == 118000 or current_iter == 120000
                or current_iter == 156000 or current_iter == 158000 or current_iter == 196000
                or current_iter == 198000 or current_iter == 234000 or current_iter == 236000):
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,
                                  param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)
            self.save_training_state(epoch, current_iter)
        else:
            pass