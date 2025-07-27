import cv2
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
import torch.fft as fft
import random
from skimage import io
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import imresize, rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class ImageNetPairedDataset_new(data.Dataset):

    def __init__(self, opt):
        super(ImageNetPairedDataset_new, self).__init__()
        self.opt = opt #导入参数
        self.scale = self.opt['scale']
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt'] # groud truth所在的文件路径
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        # self.lq_folder = opt['dataroot_lq']  # groud truth所在的文件路径
        # self.paths_lq = sorted(list(scandir(self.lq_folder, full_path=True)))

        # river_folder = '/home/ch/HAT/datasets/DF2K_tif/DF2K_HR_river'
        # self.paths_river = sorted(list(scandir(river_folder, full_path=True)))

        # self.lq_folder = opt['dataroot_lq']  # (新加的)
        #
        # if 'meta_info_file' in self.opt: #对于训练集是有此txt文件的，主要用于索引训练图片
        #     with open(self.opt['meta_info_file'], 'r') as fin:
        #         self.paths = [osp.join(self.gt_folder, line.split(' ')[0]) for line in fin]
        # else:
        #     self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))
        #
        # if 'meta_info_file' in self.opt: #对于训练集是有此txt文件的，主要用于索引训练图片
        #     with open(self.opt['meta_info_file'], 'r') as fin:
        #         self.paths_river = [osp.join(river_folder, line.split(' ')[0]) for line in fin]
        # else:
        #     self.paths_river = self.paths
        #
        # if 'meta_info_file' in self.opt: #对于训练集是有此txt文件的，主要用于索引训练图片
        #     with open(self.opt['meta_info_file'], 'r') as fin:
        #         self.paths_lq = [osp.join(self.lq_folder, line.split(' ')[0]) for line in fin]
        # else:
        #     self.paths_lq = sorted(list(scandir(self.lq_folder, full_path=True)))  # (新加的)


    def L_filter(self, x, m=0.2):
        kerl = torch.tensor(
            [[0.25 * m ** 2, 0.5 * m * (1 - m), 0.25 * m ** 2], [0.5 * m * (1 - m), (1 - m) ** 2, 0.5 * m * (1 - m)],
             [0.25 * m ** 2, 0.5 * m * (1 - m), 0.25 * m ** 2]], dtype=torch.float32)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return x_out
    def H_filter(self, x, m=0.2):
        kerl = torch.tensor(
            [[0.25 * m ** 2, -0.5 * m * (1 + m), 0.25 * m ** 2], [-0.5 * m * (1 + m), (1 + m) ** 2, -0.5 * m * (1 + m)],
             [0.25 * m ** 2, -0.5 * m * (1 + m), 0.25 * m ** 2]], dtype=torch.float32)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return x_out
    def random_horizontal_flip_opencv(self, img):
        """
        随机水平翻转图像，使用OpenCV

        参数:
        - image_path: 图像文件路径
        - probability: 翻转的概率，默认为0.5
        """
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
            # river = cv2.flip(river, type_flip)
        # 随机判断是否进行转置
        if random.uniform(0, 1) < 0.5:
            img = np.transpose(img)
            # river = np.transpose(river)
        # 随机判断是否进行高程镜像
        if random.uniform(0, 1) < 0.5:
            img = -img
            # river = -river
        return img

    def random_frequency_enhancement(self, x):
        # samples = [0.05, 0.1, 0.15]
        h, w = x.shape[-2], x.shape[-1]
        x = torch.tensor(x).view(1, h, w)
        if random.uniform(0, 1)>0.5:
            # m = random.sample(samples, 1)[0]
            m = 0.15
            if random.uniform(0, 1)>0.5:
                x = self.H_filter(x, m=m).view(h, w).numpy()
            else:
                x = self.L_filter(x, m=m).view(h, w).numpy()
        else:
            x = x.view(h, w).numpy()
        return x

    def random_add_noise(self, img):
        if random.uniform(0, 1) < 0.5:
            img = img + random.uniform(0, 1)
        return img
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
    def weight_cal(self, x):
        _, h, w = x.shape
        weight = torch.ones(1, h, w)

        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        for i in range(h):
            for j in range(w):
                s = x[:, i:i + 3, j:j + 3]
                s = s - s[0, 1, 1]
                s = torch.abs(s).sum()
                weight[0, i, j] = s
        min = torch.min(weight)
        max = torch.max(weight)
        weight = (weight - min) / (max - min + 1e-8)
        return weight
    def Gauss_filter(self, x):
        # a, b, c = 8/28, 4/28, 1/28
        # a, b, c = 0.8, 0.03, 0.02
        a, b, c = 8/20, 2/20, 1/20
        kerl = torch.tensor([[c, b, c], [b, a, b], [c, b, c]],
                            dtype=torch.float32)
        kerl = kerl.view(1, 1, 3, 3)
        x = F.pad(x, (1, 1, 1, 1), 'reflect')
        x_out = F.conv2d(input=x, weight=kerl, bias=None, stride=1, padding=0, dilation=1, groups=1)
        return x_out
    # def downsample(self, x, scale):
    #     index = random.choices([i for i in range(scale)], k=2)
    #     lr = x[index[0]::scale, index[1]::scale]
    #     return lr

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)
        # inv = self.get_inv_matrix()

        scale = self.opt['scale'] #sr与lr之比

        gt_path = self.paths[index] #加载第index个gt高程图数据的路径
        # river_path = self.paths_river[index]
        # lq_path = self.paths_lq[index]  # (新加的)

        img_gt = io.imread(gt_path) # type(img_gt): numpy.ndarray  and dtype('float32')
        # img_lq = io.imread(lq_path)
        # sr_river = io.imread(river_path)

        size_h, size_w = img_gt.shape
        if self.opt['phase'] == 'train':
            img_gt = self.random_horizontal_flip_opencv(img_gt)
            # img_gt = self.random_frequency_enhancement(img_gt)

        img_lq = cv2.resize(img_gt, (size_h // scale, size_w // scale),
                            interpolation=cv2.INTER_NEAREST) # 下采样img_gt来得到img_lq
        # img_lq = img_gt[1::3, 1::3]
        # img_lq = self.downsample(img_gt, scale)
        # sr_river = np.ascontiguousarray(sr_river, dtype=np.float32)

        sigma = 1
        #归一化
        lq_min = np.min(img_lq)
        lq_max = np.max(img_lq)
        if self.opt['phase'] == 'train':
            img_gt_low = cv2.GaussianBlur(img_gt, (5, 5), sigma, 0, sigma)
            img_gt = 2 * (img_gt - lq_min) / (lq_max - lq_min + 10) - 1
            img_lq = 2 * (img_lq - lq_min) / (lq_max - lq_min + 10) - 1
            img_gt_low = 2 * (img_gt_low - lq_min) / (lq_max - lq_min + 10) - 1
            img_gt_high = img_gt - img_gt_low
            img_gt_low = torch.tensor(img_gt_low)
            img_gt_high = torch.tensor(img_gt_high)
            img_gt_low = img_gt_low.unsqueeze(0)
            img_gt_high = img_gt_high.unsqueeze(0)
        else:
            img_gt_low = torch.ones(1, 96, 96)
            img_gt_high = torch.ones(1, 96, 96)
            img_gt = 2 * (img_gt - lq_min) / (lq_max - lq_min + 10) - 1
            img_lq = 2 * (img_lq - lq_min) / (lq_max - lq_min + 10) - 1


        # img_gt_low1 = cv2.GaussianBlur(img_gt_high, (3, 3), sigma, 0, sigma)
        img_gt = np.ascontiguousarray(img_gt, dtype=np.float32)
        img_lq = np.ascontiguousarray(img_lq, dtype=np.float32)
        # img_gt_low = np.ascontiguousarray(img_gt_low, dtype=np.float32)
        # img_gt_low1 = np.ascontiguousarray(img_gt_low1, dtype=np.float32)
        # img_gt_high = np.ascontiguousarray(img_gt_high, dtype=np.float32)

        # # river weight
        # weight = 4
        # river_weight = np.where(sr_river < 1, 1, weight * sr_river)

        # 变成tensor
        img_gt = torch.tensor(img_gt)
        img_lq = torch.tensor(img_lq)

        # if self.opt['phase'] == 'train':
        #     weight = 0.5 / (1 + torch.exp(-self.homogeneity_index(img_gt))) + 1
        #     river_weight = np.where(sr_river < 1, 1, weight * sr_river)
        #     river_weight = torch.tensor(river_weight)
        #     river_weight = river_weight.unsqueeze(0)
        # else:
        #     river_weight = torch.ones(1, 64, 64)


        # 增加维度
        img_gt = img_gt.unsqueeze(0)
        # if self.opt['phase'] == 'train':
        #     weight = self.weight_cal(img_gt)
        # else:
        #     weight = torch.ones(1, 64, 64, requires_grad=False)

        img_lq = img_lq.unsqueeze(0)


        # img_gt_low = self.Gauss_filter(img_gt)
        # img_pr = (img_gt_low + img_gt) * 0.5
        # img_gt_high = img_gt - img_gt_low
        # img_gt_low_new = self.L_filter(img_gt, m=0.2)
        # img_gt_high_new = self.H_filter(img_gt, m=0.2)
        # img_gt_low_new = self.Gauss_filter(img_gt_high)
        # img_gt_high_new = img_gt_high - img_gt_low_new
        # img_gt_low_new1 = self.Gauss_filter(img_gt_high - img_gt_low_new)
        # img_gt_high_new = self.gauss_inv(img_gt)

        return {'lq': img_lq, 'gt': img_gt , 'gt_path': gt_path
            , 'lq_path':gt_path, 'min':lq_min ,'max':lq_max, 'img_gt_low':img_gt_low,
            'img_gt_high':img_gt_high}

    def __len__(self):
        return len(self.paths)