from typing import Optional, Tuple

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from .cls_head import ClsHead
from typing import List, Optional
from mmpretrain.structures import DataSample
import torch.nn.functional as F
from ..builder import build_loss

@MODELS.register_module()
class MyLinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01)``.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss: dict,
                 init_cfg: Optional[dict] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs):
        super(MyLinearClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.compute_loss = build_loss(loss)

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def get_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        cls_score = self.fc(x)
        return cls_score

    def forward(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample]) -> torch.Tensor:
        """The forward process."""
        if isinstance(feats, tuple):
            x = feats[-1]
        losses = self.loss(x, data_samples)
        return losses

    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        targets =  torch.cat([i.gt_label for i in data_samples])
        return self.compute_loss(feats, targets)
        # return F.cross_entropy(feats, targets)
