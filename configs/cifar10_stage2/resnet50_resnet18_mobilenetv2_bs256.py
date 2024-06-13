_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10_bs16.py', 
    '../_base_/schedules/cifar10_bs128.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='KFImageClassifier',
    kd_loss=dict(type='SoftTarget',
                 temperature=10.0),
    train_cfg=dict(
        # augments=[
        #     dict(type='BatchMixup', alpha=0.1,
        #          num_classes=10, prob=0.5)
        # ],
        lambda_kd=0.1,
        lambda_feat=1.0,
        alpha=1.0,
        beta=1e-3,
        task_weight=1.0,
        teacher_checkpoint="work_dirs/resnet50_bs256/epoch_100.pth", # Input your teacher checkpoint
        feat_channels=dict(student=[128, 256, 512],
                           teacher=[512, 1024, 2048]),
    ),
    backbone=dict(
        num_task=1,
        student=dict(
            CKN=dict(type='ResNet',
                     depth=18,
                     num_stages=4,
                     out_indices=(1, 2, 3),
                     style='pytorch'),
            TSN=dict(type='TSN_backbone',
                     backbone=dict(type='MobileNetV2',
                                   out_indices=(7, ),
                                   widen_factor=0.5),
                     in_channels=1280,
                     out_channels=512)
        ),
        teacher=dict(type='ResNet',
                     depth=50,
                     num_stages=4,
                     out_indices=(1, 2, 3),
                     style='pytorch'),
    ),
    neck=dict(
        student=dict(type='GlobalAveragePooling'),
        teacher=dict(type='GlobalAveragePooling')
    ),
    head=dict(
        student=dict(
            type='MyLinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=10,
                reduction='mean',
                loss_weight=1.0),
        ),
        task=dict(
            type='MyLinearClsHead',
            num_classes=10,
            in_channels=512,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=10,
                reduction='mean',
                loss_weight=1.0),
        ),
        teacher=dict(
            type='MyLinearClsHead',
            num_classes=10,
            in_channels=2048,
            loss=dict(type='CrossEntropyLoss',
                      loss_weight=1.0),
        )
    )
)
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

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

checkpoint_config = dict(interval=20, max_keep_ckpts=2)
train_cfg = dict(max_epochs=100, val_interval=1)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=20)
    )

