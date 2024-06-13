import torch.nn as nn
import torch
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from mmpretrain.models.builder import build_backbone

@MODELS.register_module()
class TSN_backbone(BaseBackbone):
    def __init__(self, backbone, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = build_backbone(backbone)
        self.fc = nn.Linear(self.in_channels, self.out_channels, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        if isinstance(x, tuple):
            x = x[-1]

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # 论文中有介绍
        mu = torch.mean(x, 0)  # 结果均值
        log_var = torch.log(torch.var(x, 0))  # 结果方差
        return (mu, log_var), x
