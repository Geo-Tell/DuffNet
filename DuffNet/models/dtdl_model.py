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

@MODEL_REGISTRY.register()
class DTDLModel(SRModel):
    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img_lq_low, self.img_lq_high)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img_lq_low, self.img_lq_high)
    def feed_data(self, data):
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'img_lq_low' in data:
            self.img_lq_low = data['img_lq_low'].to(self.device)
        if 'img_lq_high' in data:
            self.img_lq_high = data['img_lq_high'].to(self.device)
        if 'min' in data:
            self.min = data['min'].to(self.device)
        if 'max' in data:
            self.max = data['max'].to(self.device)
        if 'gt_path' in data:
            self.path = data['gt_path']
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.img_lq_low, self.img_lq_high)

        l_total = 0
        loss_dict = OrderedDict()

        # pixel loss 重建损失
        if self.cri_pix:
            l_pix = torch.nn.functional.l1_loss(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        out_dict['min'] = self.min.detach().cpu()
        out_dict['max'] = self.max.detach().cpu()
        return out_dict
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # if (current_iter == 2000 or current_iter == 78000 or current_iter == 80000 or current_iter == 118000 or current_iter == 120000
        #         or current_iter == 156000 or current_iter == 158000 or current_iter == 196000
        #         or current_iter == 198000 or current_iter == 234000 or current_iter == 236000):
        if True:
            logger = get_root_logger()
            dataset_name = dataloader.dataset.opt['name']
            rmse_dem_sum = 0
            rmse_slope_sum = 0
            rmse_aspect_sum = 0
            j = 0
            for idx, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['gt_path'][0]))[0]
                self.feed_data(val_data)
                self.process()
                visuals = self.get_current_visuals()

                gt_tif = visuals['gt'].squeeze().numpy()  # (64,64)
                sr_tif = visuals['result'].squeeze().numpy()  # (64,64)
                h = sr_tif.shape[-1]
                max_tif = visuals['max'].numpy()
                min_tif = visuals['min'].numpy()
                sr_tif = ((sr_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                gt_tif = ((gt_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                os.makedirs(osp.join(self.opt['path']['visualization'], dataset_name), exist_ok=True)
                save_tif_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                         f'DTDL_{img_name}.TIF')

                tifffile.imwrite(save_tif_path, sr_tif)

                dem_feature = DEMFeature(torch.tensor(sr_tif).reshape(1, 1, h, h),
                                         torch.tensor(gt_tif).reshape(1, 1, h, h))
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
                torch.cuda.empty_cache()

            rmse_dem_mean = rmse_dem_sum / j
            rmse_slope_mean = rmse_slope_sum / j
            rmse_aspect_mean = rmse_aspect_sum / j

            logger.info(f'the m-rmse of dem is :{rmse_dem_mean}')
            logger.info(f'the slope-rmse of dem is :{rmse_slope_mean}')
            logger.info(f'the aspect-rmse of dem is :{rmse_aspect_mean}')
        else:
            pass
    def save(self, epoch, current_iter):
        if (current_iter==78000 or current_iter==80000 or current_iter==118000 or current_iter==120000
                or current_iter==156000 or current_iter==158000 or current_iter==196000
                or current_iter==198000 or current_iter==234000 or current_iter==236000):
        # if (current_iter == 10000 or current_iter == 15000 or current_iter == 20000
        #         or current_iter == 25000 or current_iter == 30000):
        # if (current_iter == 64000 or current_iter == 84000 or current_iter == 104000
        #         or current_iter == 124000):
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter,
                                  param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)
            self.save_training_state(epoch, current_iter)
        else:
            pass