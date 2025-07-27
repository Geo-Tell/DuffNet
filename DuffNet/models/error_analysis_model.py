import torch
from torch.nn import functional as F
import numpy as np
from collections import OrderedDict
from skimage import filters
from scipy.ndimage import uniform_filter
from scipy import stats
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
import math
import rasterio
import matplotlib.pyplot as plt
import glob
import os

@MODEL_REGISTRY.register()
class ERRORModel(SRModel):

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
        def calculate_aspect(dem, cell_size):
            # 计算梯度
            dz_dx = np.gradient(dem, cell_size, axis=1)
            dz_dy = np.gradient(dem, cell_size, axis=0)

            # 计算坡向（以度为单位）
            aspect = np.arctan2(dz_dy, -dz_dx)
            aspect = np.degrees(aspect)
            aspect = (aspect + 360) % 360  # 转换为0-360度
            return aspect
        def calculate_error(sr_dem, ref_dem):
            error = sr_dem - ref_dem
            rmse = np.sqrt(np.mean(error ** 2))
            me = np.mean(error)
            return error, rmse, me
        def classify_aspect(aspect):
            # 将坡向分为几个主要方向
            categories = {
                'North': ((270, 360), (0, 45)),
                'East': ((45, 135),),
                'South': ((135, 225),),
                'West': ((225, 270),)
            }
            classified_aspect = np.zeros_like(aspect, dtype=object)

            for category, ranges in categories.items():
                for start, end in ranges:
                    mask = (aspect >= start) & (aspect < end)
                    classified_aspect[mask] = category

            return classified_aspect
        def calculate_error_statistics(error, classified_aspect):
            categories = np.unique(classified_aspect)
            error_stats = {}
            for category in categories:
                mask = classified_aspect == category
                error_masked = error[mask]
                rmse = np.sqrt(np.mean(error_masked ** 2))
                me = np.mean(error_masked)
                error_stats[category] = {'RMSE': rmse, 'ME': me}
            return error_stats
        def process_single_sample(sr_dem, ref_dem, cell_size):
            # 计算坡向
            aspect = calculate_aspect(ref_dem, cell_size)
            # 计算误差
            error, rmse, me = calculate_error(sr_dem, ref_dem)
            # 分类坡向
            classified_aspect = classify_aspect(aspect)
            # 计算误差统计
            error_stats = calculate_error_statistics(error, classified_aspect)
            return {
                'filename': self.gt_path,
                'overall_rmse': rmse,
                'overall_me': me,
                'aspect_error_stats': error_stats,
                'error': error,
                'classified_aspect': classified_aspect
            }
        def plot_aggregated_results(all_results):
            # 汇总所有样本的误差
            all_errors = []
            all_aspects = []
            for result in all_results:
                all_errors.append(result['error'].flatten())
                all_aspects.append(result['classified_aspect'].flatten())

            all_errors = np.concatenate(all_errors)
            all_aspects = np.concatenate(all_aspects)

            # 绘制汇总的误差分布
            categories = np.unique(all_aspects)
            plt.figure(figsize=(10, 6))
            for category in categories:
                mask = all_aspects == category
                error_masked = all_errors[mask]
                if error_masked.size > 0:
                    plt.hist(error_masked, bins=30, alpha=0.5, label=category)
            plt.xlabel('Error (meters)')
            plt.ylabel('Frequency')
            plt.title('Aggregated Error Distribution by Aspect')
            plt.legend()
            plt.show()

            # 计算并打印汇总统计信息
            overall_rmse = np.sqrt(np.mean(all_errors ** 2))
            overall_me = np.mean(all_errors)
            logger.info(f"\nAggregated Overall RMSE: {overall_rmse:.2f} meters")
            logger.info(f"Aggregated Overall Mean Error: {overall_me:.2f} meters")

            # 计算按坡向分类的汇总统计
            aggregated_stats = calculate_error_statistics(all_errors, all_aspects)
            for category, stats in aggregated_stats.items():
                logger.info(f"{category}: RMSE = {stats['RMSE']:.2f} meters, ME = {stats['ME']:.2f} meters")

        cell_size = 30
        # 处理所有样本
        all_results = []
        for idx, val_data in enumerate(dataloader):
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
                    max_tif = val_data['max'].numpy()
                    min_tif = val_data['min'].numpy()
                    pred_dem = ((sr_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    gt_dem = ((gt_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
                    logger.info(f"\nProcessing {self.gt_path}...")
                    result = process_single_sample(pred_dem, gt_dem, cell_size)
                    all_results.append(result)
                if idx > 6000:
                    break
        # 绘制汇总结果
        plot_aggregated_results(all_results)






        # def compute_slope(dem):
        #     """计算坡度图（弧度制）"""
        #     dy, dx = np.gradient(dem)
        #     return np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        # def compute_roughness(dem, window_size=3):
        #     """计算地表粗糙度（滑动窗口标准差）"""
        #     return uniform_filter(dem ** 2, window_size) - uniform_filter(dem, window_size) ** 2
        #
        # all_errors = np.array([])  # 初始化为空一维数组
        # all_slopes = np.array([])
        # all_roughness = np.array([])
        #
        # if True:
        #     for idx, val_data in enumerate(dataloader):
        #         self.feed_data(val_data)
        #         self.pre_process()
        #         if 'tile' in self.opt:
        #             self.tile_process()
        #         else:
        #             self.process()
        #         self.post_process()
        #
        #         visuals = self.get_current_visuals()
        #         if self.opt['dem_tif']:
        #             # 反归一化：
        #             gt_tif = visuals['gt'].squeeze().numpy()  # (64,64)
        #             sr_tif = visuals['result'].squeeze().numpy()  # (64,64)
        #             max_tif = val_data['max'].numpy()
        #             min_tif = val_data['min'].numpy()
        #             pred_dem = ((sr_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
        #             gt_dem = ((gt_tif + 1) * 0.5) * (max_tif - min_tif + 10) + min_tif
        #             absolute_error = np.abs(gt_dem - pred_dem).flatten()
        #             slope = compute_slope(gt_dem)
        #             slope_variability = filters.gaussian(slope, sigma=1).flatten()  # 可调整sigma平滑程度
        #             roughness = compute_roughness(gt_dem).flatten()
        #
        #             all_errors = np.concatenate([all_errors, absolute_error])  # 形状: (50*64*64,)
        #             all_slopes = np.concatenate([all_slopes, slope_variability])  # 形状: (50*64*64,)
        #             all_roughness = np.concatenate([all_roughness, roughness])  # 形状: (50*64*64,)
        #
        #         # tentative for out of GPU memory
        #         del self.lq
        #         del self.output
        #         torch.cuda.empty_cache()
        #     # 计算全局相关性
        #     r_slope, p_slope = stats.pearsonr(all_errors, all_slopes)
        #     r_rough, p_rough = stats.pearsonr(all_errors, all_roughness)
        #
        #     logger.info(f'Global Correlation:')
        #     logger.info(f'Error vs Slope:       r={r_slope:.3f}, p={p_slope:.3e}')
        #     logger.info(f'Error vs Roughness:   r={r_rough:.3f}, p={p_rough:.3e}')
