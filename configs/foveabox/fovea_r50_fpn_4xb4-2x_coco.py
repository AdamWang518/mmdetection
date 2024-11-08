# _base_ = './fovea_r50_fpn_4xb4-1x_coco.py'
# # learning policy
# # max_epochs = 24
# max_epochs = 150
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[16, 22],
#         gamma=0.1)
# ]
# train_cfg = dict(max_epochs=max_epochs)
_base_ = './fovea_r50_fpn_4xb4-1x_coco.py'

# Learning policy
max_epochs = 150
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[100, 130],
        gamma=0.1
    )
]
train_cfg = dict(max_epochs=max_epochs)

# Dataset settings
ann_file_path = "D:/Github/RandomPick_v6_COCO/annotations/"
data_root = "D:/Github/RandomPick_v6_COCO/"

# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', scale=(640, 640), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(
#         type='Normalize',
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         to_rgb=True
#     ),
#     dict(type='Pad', size_divisor=32),
#     dict(type='PackDetInputs'),
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='PhotoMetricDistortion',  # 用於HSV增強
        hue_delta=5,  # 對應於YOLO的hsv_h=0.05
        saturation_range=(0.2, 1.8),  # 對應於YOLO的hsv_s=0.8
        brightness_delta=60  # 對應於YOLO的hsv_v=0.6
    ),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True
    ),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='CocoDataset',
        ann_file=ann_file_path + 'train_coco.json',
        data_prefix=dict(img='images/train/'),
        data_root=data_root,
        pipeline=train_pipeline,
        # 添加此行
        filter_cfg=dict(filter_empty_gt=False),
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        type='CocoDataset',
        ann_file=ann_file_path + 'val_coco.json',
        data_prefix=dict(img='images/val/'),
        data_root=data_root,
        pipeline=test_pipeline,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_file_path + 'val_coco.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=ann_file_path + 'test_coco.json',
    metric='bbox'
)

# 测试数据加载器可以复用验证集的配置
test_dataloader = val_dataloader

# Optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.08, momentum=0.9, weight_decay=0.0001)
)



