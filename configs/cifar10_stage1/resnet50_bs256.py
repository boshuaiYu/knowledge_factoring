_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', 
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# model config(frozen)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(3, ),  # 0, 1, 2, 3
        frozen_stages=-1,  # Stages to be frozen (stop grad and set eval mode). -1 means not freezing any parameters
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
        )
    )

# data config
# airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

train_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type={{_base_.dataset_type}},
        data_root='data/cifar10',
        split='train',
        pipeline={{_base_.train_pipeline}}),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=4,
    dataset=dict(
        type={{_base_.dataset_type}},
        data_root='data/cifar10/',
        split='test',
        pipeline={{_base_.test_pipeline}}),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

train_cfg = dict(max_epochs=100, val_interval=1)
checkpoint_config = dict(interval=20, max_keep_ckpts=2)

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10))

