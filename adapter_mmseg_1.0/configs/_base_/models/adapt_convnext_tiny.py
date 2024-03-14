norm_cfg = dict(type='SyncBN', requires_grad=True, eps=1e-6)
sem_norm_cfg = dict(type='LN2d', eps=1e-6)
checkpoint_file = \
'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-base_3rdparty_32xb128-noema_in1k_20220301-2a0ee547.pth'  # noqa
num_classes = 19
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='ConvNeXtAdapter',
        num_classes=num_classes,
        arch='tiny',
        aux_ratio=[3, 3, 3, 3],
        sem_norm_cfg=sem_norm_cfg,
        out_indices=[0, 1, 2, 3],
        num_aux_layers=[1, 1, 1, 1],
        layer_scale=[1e-5, 1e-4, 1e-3, 1e-2],
        enhance=[False] * 4,
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    decode_head=dict(
        type='AdaptSegformerHead',
        num_classes=num_classes,
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='GELU'),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
