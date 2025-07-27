import cv2
import numpy as np
import torch
import random
from skimage import io
from torch.utils import data as data

from basicsr.data.data_util import scandir
from basicsr.utils import FileClient
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class Tfa30(data.Dataset):

    def __init__(self, opt):
        super(Tfa30, self).__init__()
        self.opt = opt #导入参数
        self.scale = self.opt['scale']
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt'] # groud truth所在的文件路径
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        # river_folder = '/home/ch/HAT/datasets/DF2K_tif/DF2K_HR_river'
        # self.paths_river = sorted(list(scandir(river_folder, full_path=True)))
        self.paths_river = self.paths
    def random_horizontal_flip_opencv(self, img, river):
        random_type = random.uniform(0, 1)
        if random_type < 1 / 3:
            type_flip = 0
        elif random_type > 2 / 3:
            type_flip = -1
        else:
            type_flip = 1

        # 随机判断是否进行翻转
        if random.uniform(0, 1) < 0.5:
            img = cv2.flip(img, type_flip)  # 1表示水平翻转，0表示垂直翻转，-1表示水平和垂直同时翻转
            river = cv2.flip(river, type_flip)
        # 随机判断是否进行转置
        if random.uniform(0, 1) < 0.5:
            img = np.transpose(img)
            river = np.transpose(river)
        # 随机判断是否进行高程镜像
        if random.uniform(0, 1) < 0.5:
            img = -img
        return img, river

    def random_flip_and_mirror(self, img, river):
        if np.random.rand() > 0.5:
            img = img[:, ::-1]
            river = river[:, ::-1]
        if np.random.rand() > 0.5:
            img = img[::-1, :]
            river = river[::-1, :]
        if np.random.rand() > 0.5:
            img = img.transpose(1, 0)
            river = river.transpose(1, 0)
        # if random.uniform(0, 1) < 0.5:
        #     img = -img
        return img, river

    # 均匀性指数：
    def homogeneity_index(self, dem_data):
        rows, cols = dem_data.shape
        abs_height_diff_sum = 0
        for i in range(rows - 1):
            for j in range(cols - 1):
                abs_height_diff_sum += abs(dem_data[i, j] - dem_data[i + 1, j])
                abs_height_diff_sum += abs(dem_data[i, j] - dem_data[i, j + 1])
        homogeneity_idx = abs_height_diff_sum / (rows * cols)
        return homogeneity_idx

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale'] #sr与lr之比

        gt_path = self.paths[index] #加载第index个gt高程图数据的路径
        river_path = self.paths_river[index]

        img_gt = io.imread(gt_path) # type(img_gt): numpy.ndarray  and dtype('float32')
        sr_river = io.imread(river_path)

        size_h, size_w = img_gt.shape
        if self.opt['phase'] == 'train':
            # img_gt, sr_river = self.random_horizontal_flip_opencv(img_gt, sr_river)
            img_gt, sr_river = self.random_flip_and_mirror(img_gt, sr_river)

        img_lq = cv2.resize(img_gt, (size_h // scale, size_w // scale),
                            interpolation=cv2.INTER_NEAREST) # 下采样img_gt来得到img_lq
        sr_river = np.ascontiguousarray(sr_river, dtype=np.float32)

        #归一化
        lq_min = np.min(img_lq)
        lq_max = np.max(img_lq)
        if self.opt['phase'] == 'train':
            img_gt_low = cv2.GaussianBlur(img_gt, (7, 7), 2.5, 0, 2.5)
            img_gt = 2 * (img_gt - lq_min) / (lq_max - lq_min + 10) - 1
            img_lq = 2 * (img_lq - lq_min) / (lq_max - lq_min + 10) - 1
            img_gt_low = 2 * (img_gt_low - lq_min) / (lq_max - lq_min + 10) - 1
            img_gt_high = img_gt - img_gt_low
            img_gt_low = torch.tensor(img_gt_low)
            img_gt_high = torch.tensor(img_gt_high)
            img_gt_low = img_gt_low.unsqueeze(0)
            img_gt_high = img_gt_high.unsqueeze(0)
        else:
            img_gt_low = torch.ones(1, 64, 64)
            img_gt_high = torch.ones(1, 64, 64)
            img_gt = 2 * (img_gt - lq_min) / (lq_max - lq_min + 10) - 1
            img_lq = 2 * (img_lq - lq_min) / (lq_max - lq_min + 10) - 1


        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)
        # 变成tensor
        img_gt = torch.tensor(img_gt)
        img_lq = torch.tensor(img_lq)

        if self.opt['phase'] == 'train':
            weight = 0.5 / (1 + torch.exp(-self.homogeneity_index(img_gt))) + 1
            river_weight = np.where(sr_river < 1, 1, weight * sr_river)
            river_weight = torch.tensor(river_weight)
            river_weight = river_weight.unsqueeze(0)
        else:
            river_weight = torch.ones(1, 64, 64)


        # 增加维度
        img_gt = img_gt.unsqueeze(0)
        img_lq = img_lq.unsqueeze(0)


        return {'lq': img_lq, 'gt': img_gt , 'gt_path': gt_path, 'river_weight': river_weight
            , 'lq_path':gt_path, 'min':lq_min ,'max':lq_max, 'img_gt_low':img_gt_low,
            'img_gt_high':img_gt_high}

    def __len__(self):
        return len(self.paths)