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
class tfa30to10_dtdl(data.Dataset):

    def __init__(self, opt):
        super(tfa30to10_dtdl, self).__init__()
        self.opt = opt #导入参数
        self.scale = self.opt['scale']

        self.gt_folder = opt['dataroot_gt'] # groud truth所在的文件路径
        self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

        self.lq_folder = opt['dataroot_lq']  # groud truth所在的文件路径
        self.paths_lq = sorted(list(scandir(self.lq_folder, full_path=True)))
    def Extract_Trend_of_DEM(self, IMG, Sigma=0.5):
        WinSize = (2 * np.ceil(2 * Sigma) + 1).astype(np.int32)
        Trend = cv2.GaussianBlur(IMG, (WinSize, WinSize), Sigma, borderType=cv2.BORDER_REPLICATE)
        return Trend

    def __getitem__(self, index):
        gt_path = self.paths[index] #加载第index个gt高程图数据的路径
        lq_path = self.paths_lq[index]  # (新加的)
        img_gt = io.imread(gt_path) # type(img_gt): numpy.ndarray  and dtype('float32')
        img_lq = io.imread(lq_path)

        #归一化
        lq_min = np.min(img_lq)
        lq_max = np.max(img_lq)

        img_gt = 2 * (img_gt - lq_min) / (lq_max - lq_min + 10) - 1
        img_lq = 2 * (img_lq - lq_min) / (lq_max - lq_min + 10) - 1
        img_lq_low = self.Extract_Trend_of_DEM(img_lq)
        img_lq_high = img_lq - img_lq_low
        img_gt = torch.tensor(img_gt).unsqueeze(0)
        img_lq_low = torch.tensor(img_lq_low).unsqueeze(0)
        img_lq_high = torch.tensor(img_lq_high).unsqueeze(0)

        return {'gt': img_gt, 'min':lq_min ,'max':lq_max, 'gt_path':gt_path,
                'img_lq_low':img_lq_low, 'img_lq_high':img_lq_high}

    def __len__(self):
        return len(self.paths)