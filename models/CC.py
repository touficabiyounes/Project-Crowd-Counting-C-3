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
            self.CCN = torch.nn.DataParallel(self.CCN, device_ids=gpus).to(device)
        else:
            self.CCN = self.CCN.to(device)

        # Which loss to use (set in config.py via cfg.LOSS)
        self.loss_type = cfg.LOSS if hasattr(cfg, 'LOSS') else 'mse'

        self.loss_mse_fn = nn.MSELoss().to(device)

        if self.loss_type in ('gaussian_nll', 'laplace_nll'):
            self.log_var_head = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, kernel_size=1),
            ).to(device)
            # Initialise with near-zero weights so training starts close to MSE behavior
            for m in self.log_var_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    @property
    def loss(self):
        return self.loss_mse

    def forward(self, img, gt_map):
        density_map = self.CCN(img) # [B, 1, H, W]
        self.loss_mse = self.build_loss(density_map, gt_map)
        return density_map

    def build_loss(self, density_map, gt_data):
        """
        Three loss modes:

        'mse'          – standard pixel-wise MSE, identical to the original code.
        'gaussian_nll' – heteroscedastic Gaussian NLL from Kendall & Gal (2017).
                         The network additionally predicts s = log σ² per pixel.
                         L = 0.5 * exp(-s) * (μ - y)² + 0.5 * s
        'laplace_nll'  – heteroscedastic Laplace NLL from Kendall & Gal (2017).
                         s = log b (log scale of the Laplace distribution).
                         L = exp(-s) * |μ - y| + s
        """
        if self.loss_type == 'mse':
            return self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())

        log_var = self.log_var_head(density_map.detach())
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        self.log_var_map = log_var.detach()

        diff = density_map - gt_data

        if self.loss_type == 'gaussian_nll':
            loss = 0.5 * torch.exp(-log_var) * diff.pow(2) + 0.5 * log_var

        elif self.loss_type == 'laplace_nll':
            loss = torch.exp(-log_var) * diff.abs() + log_var

        aux_mse = self.loss_mse_fn(density_map.squeeze(), gt_data.squeeze())
        return loss.mean() + 0.5 * aux_mse

    def test_forward(self, img):
        density_map = self.CCN(img)
        return density_map