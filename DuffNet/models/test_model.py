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
class TESTModel(SRModel):

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
        if 'gt_path' in data:
            self.path = data['gt_path']
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output1, self.output2= self.net_g(self.lq)
        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix2 = torch.nn.functional.l1_loss(self.output2, self.gt)
            l_total += l_pix2
            loss_dict['l_pix2'] = l_pix2

        # output1的重建损失
        if self.cri_pix:
            l_pix1_0 = torch.nn.functional.l1_loss(self.output1[0]*self.river_weight, self.img_gt_low*self.river_weight)
            l_pix1_1 = torch.nn.functional.l1_loss(self.output1[1]*self.river_weight, self.img_gt_high*self.river_weight)
            l_total += l_pix1_0
            loss_dict['l_pix1_0'] = l_pix1_0
            l_total += l_pix1_1
            loss_dict['l_pix1_1'] = l_pix1_1
        l_total.backward()
        self.optimizer_g.step()
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

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        dataset_name = dataloader.dataset.opt['name']
        if True:
            rmse_dem_sum = 0
            rmse_slope_sum = 0
            rmse_aspect_sum = 0
            j = 0
            for idx, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
                self.feed_data(val_data)

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
                    h = sr_tif.shape[-1]
                    max_tif = val_data['max'].numpy()
                    min_tif = val_data['min'].numpy()

                    sr_tif = ((sr_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name), exist_ok=True)
                    save_tif_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'DuffNet_{img_name}.TIF')

                    tifffile.imwrite(save_tif_path, sr_tif)
                    gt_tif = ((gt_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    dem_feature = DEMFeature(torch.tensor(sr_tif).reshape(1,1,h,h),torch.tensor(gt_tif).reshape(1,1,h,h))
                    slope = dem_feature['slope_1_rmse']
                    aspect = dem_feature['aspect_1_rmse']
                    # 计算平方误差
                    squared_error = (sr_tif - gt_tif) ** 2
                    mean_squared_error = np.mean(squared_error)
                    rmse_dem = np.sqrt(mean_squared_error)

                    result_log = (f"{self.path}:"
                                  f"rmse: {rmse_dem:.4f}; "
                                  f"slope: {slope:.4f}; "
                                  f"aspect: {aspect:.4f}")
                    logger.info(result_log)

                    rmse_dem_sum = rmse_dem + rmse_dem_sum
                    rmse_slope_sum = slope + rmse_slope_sum
                    rmse_aspect_sum = aspect + rmse_aspect_sum
                    j = j + 1

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

            rmse_dem_mean = rmse_dem_sum / j
            rmse_slope_mean = rmse_slope_sum / j
            rmse_aspect_mean = rmse_aspect_sum / j
            logger.info(f'the m-rmse of dem is :{rmse_dem_mean}')
            logger.info(f'the slope-rmse of dem is :{rmse_slope_mean}')
            logger.info(f'the aspect-rmse of dem is :{rmse_aspect_mean}')

    def save(self, epoch, current_iter):
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