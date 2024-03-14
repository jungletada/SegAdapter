# model settings
sem_norm_cfg = dict(type='LN2d', requires_grad=True, eps=1e-6)
norm_cfg = dict(type='SyncBN', requires_grad=True)
num_classes = 19

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='pretrain/mit_b0.pth',
    backbone=dict(
        type='SegformerSeMask',
        num_classes=num_classes,
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=[0, 1, 2, 3],
        mlp_ratio=4,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        qkv_bias=True),
    decode_head=dict(
        type='AdaptSegformerHead',
        num_classes=num_classes,
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        base_channels=256,
        channels=256,
        dropout_ratio=0.1,
        cls_kernel_size=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='GELU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
