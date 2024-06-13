_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', 
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# model config(frozen)
model = dict(
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3, ),  # 0, 1, 2, 3
        style="pytorch"
    ),
    
    neck=dict(type='GlobalAveragePooling'),

    head=dict(
        type='LinearClsHead',
        num_classes=10,
        in_channels=2048,
        loss=dict(
            type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
        ),
    )

# load_from = "work_dirs/resnet50_bs256/epoch_100.pth"

test_dataloader = dict(DataLoader(batch_size=32,
                            shuffle=False,
                            dataset=torchvision.datasets.SVHN(
                                'data/svhn',
                                split="test",
                                download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(**norm_cfg)
                                ]))))
