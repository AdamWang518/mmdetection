_base_ = './fovea_r50_fpn_4xb4-1x_coco.py'
# learning policy
max_epochs = 24
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[16, 22],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs)