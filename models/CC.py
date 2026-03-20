import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from misc.utils import device
import pdb

class CrowdCounter(nn.Module):
    def __init__(self, gpus, model_name):
        super(CrowdCounter, self).__init__()

        if model_name == 'AlexNet':
            from .SCC_Model.AlexNet import AlexNet as net
        elif model_name == 'VGG':
            from .SCC_Model.VGG import VGG as net
        elif model_name == 'VGG_DECODER':
            from .SCC_Model.VGG_decoder import VGG_decoder as net
        elif model_name == 'MCNN':
            from .SCC_Model.MCNN import MCNN as net
        elif model_name == 'CSRNet':
            from .SCC_Model.CSRNet import CSRNet as net
        elif model_name == 'Res50':
            from .SCC_Model.Res50 import Res50 as net
        elif model_name == 'Res101':
            from .SCC_Model.Res101 import Res101 as net
        elif model_name == 'Res101_SFCN':
            from .SCC_Model.Res101_SFCN import Res101_SFCN as net

        self.CCN = net()
        if len(gpus) > 1 and device.type == 'cuda':
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus)
        self.CCN = self.CCN.to(device)
        self.loss_mse_fn = nn.MSELoss().to(device)

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        output = self.CCN(img)

        if isinstance(output, tuple):
            mu, b = output
            if cfg.LOSS_TYPE == 'laplace':
                self.loss_mse = self.build_laplace_loss(mu.squeeze(), b.squeeze(), gt_map.squeeze())
            elif cfg.LOSS_TYPE == 'gaussian':
                self.loss_mse = self.build_gaussian_loss(mu.squeeze(), b.squeeze(), gt_map.squeeze())
            else:
                self.loss_mse = self.build_loss(mu.squeeze(), gt_map.squeeze())
            return mu
        else:
            density_map = output
            self.loss_mse = self.build_loss(density_map.squeeze(), gt_map.squeeze())
            return density_map

    def build_loss(self, density_map, gt_data):
        return self.loss_mse_fn(density_map, gt_data)

    def build_laplace_loss(self, mu, b, gt_data):
        laplace = torch.mean(torch.log(b) + torch.abs(gt_data - mu) / b)
        count_loss = torch.abs(mu.sum() - gt_data.sum()) / (gt_data.sum() + 1)
        return laplace + count_loss

    def build_gaussian_loss(self, mu, b, gt_data):
        gaussian = nn.GaussianNLLLoss(reduction='mean')(mu, gt_data, b)
        count_loss = torch.abs(mu.sum() - gt_data.sum()) / (gt_data.sum() + 1)
        return gaussian + count_loss

    def test_forward(self, img):
        output = self.CCN(img)
        if isinstance(output, tuple):
            mu, _ = output
            return mu
        return output