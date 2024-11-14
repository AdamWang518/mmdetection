_base_ = [
    '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'  # Faster R-CNN的基礎配置
]

model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='path/to/swin_tiny_patch4_window7_224.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5
    )
    # `rpn_head` 和 `roi_head` 將自動從基礎配置中繼承
)

# 其他設定，確保與Swin配置一致
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.05)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr_ratio=1e-5,
)
runner = dict(type='EpochBasedRunner', max_epochs=12)
