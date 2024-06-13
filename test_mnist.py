# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from copy import deepcopy

import mmengine
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from torch.nn.modules import Module
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from mmengine.model import BaseModel
from mmpretrain.models.builder import build_backbone, build_neck, build_loss, build_head
norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])


def load_ckpt(model, ckpt):
        state_dict = torch.load(ckpt)
        model.load_state_dict(state_dict)

        for param in model.parameters():
            param.requires_grad = False

class TestModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config["backbone"])
        self.neck = build_neck(config["neck"])
        self.head = build_head(config["head"])
    
    def forward(self, inputs, labels):
        return inputs, labels

def main(config):
    # load config
    cfg = Config.fromfile(config)
    
    
    test_dataloader = DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.MNIST(
                                'data/mnist',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ])))
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # if args.out and args.out_item in ['pred', None]:
    #     runner.test_evaluator.metrics.append(
    #         DumpResults(out_file_path=args.out))

    # start testing
    metrics = runner.test()

    # if args.out and args.out_item == 'metrics':
    #     mmengine.dump(metrics, args.out)


if __name__ == '__main__':
    main("configs/mnist_teacher_test/mnist_teacher_test1.py")
