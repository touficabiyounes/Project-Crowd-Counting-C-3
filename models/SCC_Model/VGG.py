from config import cfg
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *

# model_path = '../PyTorch_Pretrained/vgg16-397923af.pth'

class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        vgg = models.vgg16(pretrained=pretrained)
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])
        self.use_uncertainty = cfg.LOSS in ['gaussian', 'laplace']

        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))

        if self.use_uncertainty:
            self.log_var = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                         Conv2d(128, 1, 1, same_padding=True, NL=None))

    def forward(self, x):
        x = self.features4(x)
        mu = self.de_pred(x)
        mu = F.interpolate(mu, scale_factor=8)

        if not self.use_uncertainty:
            return mu
        
        log_b = self.log_var(x)
        log_b = torch.clamp(log_b, min=-7.0, max=2.0)
        b = torch.exp(log_b)
        b = F.interpolate(b, scale_factor=8)
        return mu, b